"""In-process job manager for read-merge (consensus) jobs.

A merge job consumes the two published gzip outputs of a *successful* QC job
and produces one single-end FASTQ.gz.  The concurrency contract is identical
to the QC manager's:

* a stop requested while pending/running atomically moves the job to
  ``stopping`` and sets its event; it can never subsequently become
  ``succeeded``;
* a job that already reached ``succeeded``/``failed``/``stopped`` rejects
  stop requests (the API maps that to HTTP 409);
* ``stopping`` is only turned into ``stopped`` after the worker has exited
  and its temporary files have been removed;
* the merged output is written to a ``.part`` file and atomically renamed
  only on success, so a corrupt input gzip, a write failure, or an accepted
  stop never publishes a half-written result.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .jobs import (
    FAILED,
    PENDING,
    RUNNING,
    STOPPABLE_STATUSES,
    STOPPED,
    STOPPING,
    SUCCEEDED,
    JobConflict,
    JobManager,
    JobNotFound,
    _now,
)
from .merge import MergeResult, run_merge
from .qc import JobStopped, QCError

logger = logging.getLogger(__name__)


@dataclass
class MergeJob:
    merge_id: str
    directory: Path
    source_job_id: str
    source_r1: Path
    source_r2: Path
    status: str = PENDING
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    pairs_processed: int = 0
    total_pairs: int | None = None
    merged_pairs: int | None = None
    no_overlap_pairs: int | None = None
    ambiguous_pairs: int | None = None
    non_acgt_pairs: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None

    # -- paths -----------------------------------------------------------
    @property
    def part_output(self) -> Path:
        return self.directory / "merged.fastq.gz.part"

    @property
    def output(self) -> Path:
        return self.directory / "merged.fastq.gz"

    @property
    def download_ready(self) -> bool:
        return self.status == SUCCEEDED and self.output.is_file()

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "merge_id": self.merge_id,
            "source_job_id": self.source_job_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pairs_processed": self.pairs_processed,
            "total_pairs": self.total_pairs,
            "merged_pairs": self.merged_pairs,
            "no_overlap_pairs": self.no_overlap_pairs,
            "ambiguous_pairs": self.ambiguous_pairs,
            "non_acgt_pairs": self.non_acgt_pairs,
            "error": (
                {"code": self.error_code, "message": self.error_message}
                if self.error_code
                else None
            ),
            "downloads": None,
        }
        if self.download_ready:
            result["downloads"] = {
                "merged": f"/merge-jobs/{self.merge_id}/download",
            }
        return result


class MergeJobManager:
    def __init__(self, data_dir: Path, qc_manager: JobManager) -> None:
        self.data_dir = Path(data_dir)
        self.merge_dir = self.data_dir / "merge-jobs"
        self.merge_dir.mkdir(parents=True, exist_ok=True)
        self._qc_manager = qc_manager
        self._jobs: dict[str, MergeJob] = {}
        self._lock = threading.Lock()
        self._remove_stale_directories()

    # -- lifecycle -------------------------------------------------------
    def _remove_stale_directories(self) -> None:
        """Drop directories left behind by a previous process."""
        for child in self.merge_dir.iterdir():
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort
                logger.warning("could not remove stale path %s", child)

    def create_job(self, source_job_id: str) -> MergeJob:
        """Create a merge job over a successful QC job's published outputs.

        Raises ``JobNotFound`` when the QC job is unknown and ``JobConflict``
        when it has not successfully completed (the API maps these to 404
        and 409).
        """
        source_r1, source_r2 = self._qc_manager.output_paths(source_job_id)
        merge_id = uuid.uuid4().hex
        directory = self.merge_dir / merge_id
        directory.mkdir(parents=True)
        job = MergeJob(
            merge_id=merge_id,
            directory=directory,
            source_job_id=source_job_id,
            source_r1=source_r1,
            source_r2=source_r2,
        )
        with self._lock:
            self._jobs[merge_id] = job
        return job

    def start(self, merge_id: str) -> None:
        job = self.get(merge_id)
        thread = threading.Thread(
            target=self._run_job,
            args=(job,),
            name=f"merge-{merge_id[:8]}",
            daemon=True,
        )
        with self._lock:
            if job.status != PENDING:
                raise JobConflict("merge job has already been started")
            job.thread = thread
        thread.start()

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.stop_event.set()
        for job in jobs:
            thread = job.thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)

    # -- queries ---------------------------------------------------------
    def get(self, merge_id: str) -> MergeJob:
        with self._lock:
            job = self._jobs.get(merge_id)
        if job is None:
            raise JobNotFound(merge_id)
        return job

    def list_jobs(self) -> list[MergeJob]:
        with self._lock:
            jobs = list(self._jobs.values())
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def snapshot(self, merge_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(merge_id)
            if job is None:
                raise JobNotFound(merge_id)
            return job.to_dict()

    def request_stop(self, merge_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(merge_id)
            if job is None:
                raise JobNotFound(merge_id)
            if job.status in STOPPABLE_STATUSES:
                job.status = STOPPING
                job.stop_event.set()
            elif job.status == STOPPING:
                job.stop_event.set()
            elif job.status == STOPPED:
                pass  # idempotent
            else:  # succeeded / failed
                raise JobConflict(
                    f"merge job is already {job.status}; it can no longer "
                    "be stopped"
                )
            return job.to_dict()

    def output_path(self, merge_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(merge_id)
            if job is None:
                raise JobNotFound(merge_id)
            if job.status != SUCCEEDED:
                raise JobConflict(
                    f"merge job status is {job.status!r}; the download is "
                    "only available after successful completion"
                )
            if not job.output.is_file():
                raise JobConflict("result file is no longer available")
            return job.output

    # -- worker ----------------------------------------------------------
    def _run_job(self, job: MergeJob) -> None:
        with self._lock:
            if job.status == STOPPING or job.stop_event.is_set():
                # A stop request landed while the worker was still starting.
                stop_early = True
            elif job.status == PENDING:
                job.status = RUNNING
                job.started_at = _now()
                stop_early = False
            else:  # pragma: no cover - defensive
                return
        if stop_early:
            self._finish_stopped(job)
            return

        try:
            result = run_merge(
                job.source_r1,
                job.source_r2,
                job.part_output,
                job.stop_event,
                progress_cb=lambda pairs: self._set_progress(job, pairs),
            )
        except JobStopped:
            self._finish_stopped(job)
        except QCError as exc:
            self._finish_failed(job, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - unexpected bug
            logger.exception("unexpected error in merge job %s", job.merge_id)
            self._finish_failed(job, "internal_error", str(exc))
        else:
            self._finish_completed(job, result)

    def _set_progress(self, job: MergeJob, pairs: int) -> None:
        with self._lock:
            if pairs > job.pairs_processed:
                job.pairs_processed = pairs

    def _finish_completed(self, job: MergeJob, result: MergeResult) -> None:
        with self._lock:
            if job.status == STOPPING or job.stop_event.is_set():
                # A stop was accepted while the worker was finishing.
                stop_wins = True
            else:
                stop_wins = False
                # Publish under the same lock used by stop requests and
                # download lookups: once ``succeeded`` is visible the file is
                # already in place, and a later stop is rejected.
                os.replace(job.part_output, job.output)
                job.total_pairs = result.total_pairs
                job.merged_pairs = result.merged_pairs
                job.no_overlap_pairs = result.no_overlap_pairs
                job.ambiguous_pairs = result.ambiguous_pairs
                job.non_acgt_pairs = result.non_acgt_pairs
                job.pairs_processed = result.total_pairs
                job.finished_at = _now()
                job.status = SUCCEEDED
        if stop_wins:
            self._finish_stopped(job)

    def _finish_failed(self, job: MergeJob, code: str, message: str) -> None:
        with self._lock:
            if job.status == STOPPING or job.stop_event.is_set():
                stop_wins = True
            else:
                stop_wins = False
                job.status = FAILED
                job.error_code = code
                job.error_message = message
                job.finished_at = _now()
        if stop_wins:
            self._finish_stopped(job)
            return
        self._cleanup_all(job)

    def _finish_stopped(self, job: MergeJob) -> None:
        # Temporary files are removed *before* the job is reported as
        # stopped, as required by the stop contract.
        self._cleanup_all(job)
        with self._lock:
            job.total_pairs = None
            job.merged_pairs = None
            job.no_overlap_pairs = None
            job.ambiguous_pairs = None
            job.non_acgt_pairs = None
            job.error_code = None
            job.error_message = None
            job.finished_at = _now()
            job.status = STOPPED

    def _cleanup_all(self, job: MergeJob) -> None:
        shutil.rmtree(job.directory, ignore_errors=True)
