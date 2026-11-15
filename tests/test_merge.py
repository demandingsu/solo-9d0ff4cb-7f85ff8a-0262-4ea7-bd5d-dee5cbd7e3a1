"""Tests for the merge (consensus) job API."""

from __future__ import annotations

import gzip
import io
import random
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def fastq_record(name: str, seq: str, qual: str) -> str:
    return f"@{name}\n{seq}\n+\n{qual}\n"


def gz_bytes(text: str) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(text.encode("ascii"))
    return buffer.getvalue()


def read_gz(data: bytes) -> str:
    return gzip.decompress(data).decode("ascii")


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


HIGH = "I"  # Phred 40


@pytest.fixture()
def client(tmp_path: Path):
    app = create_app(data_dir=tmp_path / "data")
    with TestClient(app) as test_client:
        yield test_client


def wait_for(client: TestClient, url: str, statuses, timeout=15.0) -> dict:
    deadline = time.monotonic() + timeout
    payload = {}
    while time.monotonic() < deadline:
        payload = client.get(url).json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"{url} did not reach {statuses}: {payload}")


def run_qc_job(client: TestClient, r1: str, r2: str, threshold: int = 20) -> str:
    """Submit a QC job and return its id once it succeeded."""
    response = client.post(
        "/jobs",
        files={
            "r1": ("R1.fastq.gz", gz_bytes(r1), "application/gzip"),
            "r2": ("R2.fastq.gz", gz_bytes(r2), "application/gzip"),
        },
        data={"min_quality": str(threshold)},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    done = wait_for(client, f"/jobs/{job_id}", {"succeeded", "failed"})
    assert done["status"] == "succeeded", done
    return job_id


def submit_merge(client: TestClient, job_id: str):
    return client.post("/merge-jobs", data={"job_id": job_id})


def run_merge_job(client: TestClient, job_id: str) -> dict:
    response = submit_merge(client, job_id)
    assert response.status_code == 202, response.text
    merge_id = response.json()["merge_id"]
    return wait_for(
        client, f"/merge-jobs/{merge_id}", {"succeeded", "failed", "stopped"}
    )


def parse_fastq(text: str) -> list[tuple[str, str, str]]:
    lines = text.splitlines()
    assert len(lines) % 4 == 0
    records = []
    for i in range(0, len(lines), 4):
        assert lines[i].startswith("@")
        assert lines[i + 2].startswith("+")
        records.append((lines[i][1:], lines[i + 1], lines[i + 3]))
    return records


# ---------------------------------------------------------------------------
# Happy path and merge rules
# ---------------------------------------------------------------------------


def test_merge_happy_path_roundtrip(client):
    rng = random.Random(7)
    fragments = [rand_seq(rng, 120) for _ in range(3)]
    r1 = "".join(
        fastq_record(f"read{i}/1", frag[:75], HIGH * 75)
        for i, frag in enumerate(fragments)
    )
    r2 = "".join(
        fastq_record(f"read{i}/2", revcomp(frag[45:]), HIGH * 75)
        for i, frag in enumerate(fragments)
    )
    job_id = run_qc_job(client, r1, r2)

    done = run_merge_job(client, job_id)
    assert done["status"] == "succeeded", done
    assert done["source_job_id"] == job_id
    assert done["total_pairs"] == 3
    assert done["merged_pairs"] == 3
    assert done["no_overlap_pairs"] == 0
    assert done["ambiguous_pairs"] == 0
    assert done["non_acgt_pairs"] == 0
    assert done["downloads"] == {
        "merged": f"/merge-jobs/{done['merge_id']}/download"
    }

    response = client.get(done["downloads"]["merged"])
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/gzip"
    records = parse_fastq(read_gz(response.content))
    # Names keep the original order and lose the /1 suffix.
    assert [name for name, _, _ in records] == ["read0", "read1", "read2"]
    # Each consensus read reconstructs the full fragment; the overlap takes
    # the higher quality (all 'I' here).
    for (name, seq, qual), frag in zip(records, fragments):
        assert seq == frag
        assert qual == HIGH * len(frag)


def test_merge_consensus_quality_rules(client):
    rng = random.Random(11)
    frag = rand_seq(rng, 120)
    r1_seq = frag[:75]
    r2rc = list(frag[45:])
    # One mismatching position inside the 30-base overlap (R1 coord 55).
    pos = 10
    original = r2rc[pos]
    r2rc[pos] = "A" if original != "A" else "C"
    r2_seq = revcomp("".join(r2rc))

    # Case A: quality delta >= 5 -> higher-quality end (R1) wins.
    r1 = fastq_record("a/1", r1_seq, HIGH * 75)
    r2 = fastq_record("a/2", r2_seq, "5" * 75)  # Phred 20, delta 20
    job_id = run_qc_job(client, r1, r2, threshold=0)
    done = run_merge_job(client, job_id)
    assert done["merged_pairs"] == 1, done
    records = parse_fastq(read_gz(client.get(done["downloads"]["merged"]).content))
    name, seq, qual = records[0]
    assert name == "a"
    assert seq == frag  # R1 base kept at the mismatch
    assert qual[55] == HIGH  # winner keeps its own quality

    # Case B: quality delta < 5 -> N with lowest quality '!'.
    r1 = fastq_record("b/1", r1_seq, "5" * 75)  # Phred 20
    r2 = fastq_record("b/2", r2_seq, "7" * 75)  # Phred 22, delta 2
    job_id = run_qc_job(client, r1, r2, threshold=0)
    done = run_merge_job(client, job_id)
    assert done["merged_pairs"] == 1, done
    records = parse_fastq(read_gz(client.get(done["downloads"]["merged"]).content))
    _, seq, qual = records[0]
    assert seq[:55] == frag[:55] and seq[56:] == frag[56:]
    assert seq[55] == "N"
    assert qual[55] == "!"


def test_merge_counts_each_skip_reason(client):
    rng = random.Random(13)
    # Pair 1: mergeable. Pair 2: no overlap (fragment longer than both reads).
    # Pair 3: ambiguous (homopolymer). Pair 4: contains a non-ACGT base.
    frag1 = rand_seq(rng, 120)
    frag2 = rand_seq(rng, 200)
    r1 = (
        fastq_record("ok/1", frag1[:75], HIGH * 75)
        + fastq_record("far/1", frag2[:75], HIGH * 75)
        + fastq_record("poly/1", "A" * 60, HIGH * 60)
        + fastq_record("amb/1", "A" * 30 + "N" + "A" * 29, HIGH * 60)
    )
    r2 = (
        fastq_record("ok/2", revcomp(frag1[45:]), HIGH * 75)
        + fastq_record("far/2", revcomp(frag2[125:]), HIGH * 75)
        + fastq_record("poly/2", "T" * 60, HIGH * 60)
        + fastq_record("amb/2", "T" * 60, HIGH * 60)
    )
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    assert done["status"] == "succeeded", done
    assert done["total_pairs"] == 4
    assert done["merged_pairs"] == 1
    assert done["no_overlap_pairs"] == 1
    assert done["ambiguous_pairs"] == 1
    assert done["non_acgt_pairs"] == 1

    records = parse_fastq(read_gz(client.get(done["downloads"]["merged"]).content))
    assert [(name,) for name, _, _ in records] == [("ok",)]
    assert records[0][1] == frag1


def test_merge_two_mismatches_not_merged(client):
    rng = random.Random(17)
    frag = rand_seq(rng, 120)
    r1_seq = frag[:75]
    r2rc = list(frag[45:])
    for pos in (10, 20):
        original = r2rc[pos]
        r2rc[pos] = "A" if original != "A" else "C"
    r1 = fastq_record("x/1", r1_seq, HIGH * 75)
    r2 = fastq_record("x/2", revcomp("".join(r2rc)), HIGH * 75)
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    assert done["merged_pairs"] == 0
    assert done["no_overlap_pairs"] == 1


def test_merge_overlap_boundary(client):
    rng = random.Random(19)
    frag = rand_seq(rng, 120)
    # Overlap exactly 20 -> merged.
    r1 = fastq_record("edge/1", frag[:70], HIGH * 70)
    r2 = fastq_record("edge/2", revcomp(frag[50:]), HIGH * 70)
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    assert done["merged_pairs"] == 1, done
    records = parse_fastq(read_gz(client.get(done["downloads"]["merged"]).content))
    assert records[0][1] == frag

    # Overlap 19 -> not merged.
    r1 = fastq_record("short/1", frag[:70], HIGH * 70)
    r2 = fastq_record("short/2", revcomp(frag[51:]), HIGH * 69)
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    assert done["merged_pairs"] == 0
    assert done["no_overlap_pairs"] == 1


def test_merge_preserves_pair_order(client):
    rng = random.Random(23)
    fragments = [rand_seq(rng, 110) for _ in range(6)]
    r1 = "".join(
        fastq_record(f"r{i}/1", frag[:70], HIGH * 70)
        for i, frag in enumerate(fragments)
    )
    r2 = "".join(
        fastq_record(f"r{i}/2", revcomp(frag[40:]), HIGH * 70)
        for i, frag in enumerate(fragments)
    )
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    records = parse_fastq(read_gz(client.get(done["downloads"]["merged"]).content))
    assert [name for name, _, _ in records] == [f"r{i}" for i in range(6)]
    for (_, seq, _), frag in zip(records, fragments):
        assert seq == frag


def test_merge_list_jobs(client):
    rng = random.Random(29)
    frag = rand_seq(rng, 120)
    r1 = fastq_record("a/1", frag[:75], HIGH * 75)
    r2 = fastq_record("a/2", revcomp(frag[45:]), HIGH * 75)
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    listing = client.get("/merge-jobs").json()["jobs"]
    assert [job["merge_id"] for job in listing] == [done["merge_id"]]
    assert listing[0]["source_job_id"] == job_id


# ---------------------------------------------------------------------------
# Source-job validation
# ---------------------------------------------------------------------------


def test_merge_unknown_source_404(client):
    response = submit_merge(client, "no-such-job")
    assert response.status_code == 404


def test_merge_requires_succeeded_source(client):
    # A failed QC job cannot be merged (name mismatch makes QC fail).
    r1 = fastq_record("a/1", "ACGT", HIGH * 4)
    r2 = fastq_record("b/2", "ACGT", HIGH * 4)
    response = client.post(
        "/jobs",
        files={
            "r1": ("R1.fastq.gz", gz_bytes(r1), "application/gzip"),
            "r2": ("R2.fastq.gz", gz_bytes(r2), "application/gzip"),
        },
        data={"min_quality": "20"},
    )
    job_id = response.json()["job_id"]
    done = wait_for(client, f"/jobs/{job_id}", {"failed"})
    assert done["status"] == "failed"
    response = submit_merge(client, job_id)
    assert response.status_code == 409


def test_merge_rejects_running_source(client, monkeypatch):
    import app.jobs as jobs_module

    release = threading.Event()
    entered = threading.Event()
    real_run_qc = jobs_module.run_qc

    def blocking_run_qc(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_run_qc(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "run_qc", blocking_run_qc)
    r1 = fastq_record("a/1", "ACGT", HIGH * 4)
    r2 = fastq_record("a/2", "ACGT", HIGH * 4)
    response = client.post(
        "/jobs",
        files={
            "r1": ("R1.fastq.gz", gz_bytes(r1), "application/gzip"),
            "r2": ("R2.fastq.gz", gz_bytes(r2), "application/gzip"),
        },
        data={"min_quality": "20"},
    )
    job_id = response.json()["job_id"]
    assert entered.wait(10)
    assert submit_merge(client, job_id).status_code == 409
    release.set()
    wait_for(client, f"/jobs/{job_id}", {"succeeded"})


def test_merge_unknown_merge_id_404(client):
    assert client.get("/merge-jobs/nope").status_code == 404
    assert client.post("/merge-jobs/nope/stop").status_code == 404
    assert client.get("/merge-jobs/nope/download").status_code == 404


# ---------------------------------------------------------------------------
# Failure handling: corrupt inputs, no half-written outputs
# ---------------------------------------------------------------------------


def test_merge_fails_on_corrupt_qc_output(client, tmp_path):
    rng = random.Random(31)
    frag = rand_seq(rng, 120)
    r1 = fastq_record("a/1", frag[:75], HIGH * 75)
    r2 = fastq_record("a/2", revcomp(frag[45:]), HIGH * 75)
    job_id = run_qc_job(client, r1, r2)

    # Corrupt the published R2 gzip (truncate the stream) after QC succeeded.
    manager = client.app.state.manager
    r2_path = manager.jobs_dir / job_id / "R2.kept.fastq.gz"
    data = r2_path.read_bytes()
    r2_path.write_bytes(data[:-6])

    done = run_merge_job(client, job_id)
    assert done["status"] == "failed", done
    assert done["error"]["code"] == "gzip_corrupt"
    assert done["merged_pairs"] is None
    assert client.get(done["downloads"]["merged"] if done["downloads"] else f"/merge-jobs/{done['merge_id']}/download").status_code == 409
    # No partial output may be left behind.
    merge_dir = client.app.state.merge_manager.merge_dir / done["merge_id"]
    assert not merge_dir.exists()


def test_merge_fails_on_pair_count_mismatch(client):
    rng = random.Random(37)
    frag = rand_seq(rng, 120)
    r1 = fastq_record("a/1", frag[:75], HIGH * 75)
    r2 = fastq_record("a/2", revcomp(frag[45:]), HIGH * 75)
    job_id = run_qc_job(client, r1, r2)

    # Append an extra record to the published R1 output.
    manager = client.app.state.manager
    r1_path = manager.jobs_dir / job_id / "R1.kept.fastq.gz"
    extra = gz_bytes(fastq_record("extra/1", "ACGT", HIGH * 4))
    r1_path.write_bytes(r1_path.read_bytes() + extra)

    done = run_merge_job(client, job_id)
    assert done["status"] == "failed", done
    assert done["error"]["code"] == "pair_count_mismatch"
    merge_dir = client.app.state.merge_manager.merge_dir / done["merge_id"]
    assert not merge_dir.exists()


def test_merge_empty_input_succeeds_with_zero_counts(client):
    # A QC job that keeps nothing publishes empty (but valid) gzip files.
    r1 = fastq_record("a/1", "ACGT", "####")
    r2 = fastq_record("a/2", "ACGT", "####")
    job_id = run_qc_job(client, r1, r2, threshold=20)
    done = run_merge_job(client, job_id)
    assert done["status"] == "succeeded", done
    assert done["total_pairs"] == 0
    assert done["merged_pairs"] == 0
    body = client.get(done["downloads"]["merged"]).content
    assert read_gz(body) == ""


# ---------------------------------------------------------------------------
# Stop semantics
# ---------------------------------------------------------------------------


def _blocked_merge(client, monkeypatch):
    """Start a merge job whose engine blocks until released."""
    import app.merge_jobs as merge_jobs_module

    release = threading.Event()
    entered = threading.Event()
    real_run_merge = merge_jobs_module.run_merge

    def blocking_run_merge(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return real_run_merge(*args, **kwargs)

    monkeypatch.setattr(merge_jobs_module, "run_merge", blocking_run_merge)

    rng = random.Random(41)
    frag = rand_seq(rng, 120)
    r1 = fastq_record("a/1", frag[:75], HIGH * 75)
    r2 = fastq_record("a/2", revcomp(frag[45:]), HIGH * 75)
    job_id = run_qc_job(client, r1, r2)
    merge_id = submit_merge(client, job_id).json()["merge_id"]
    assert entered.wait(10)
    return merge_id, release


def test_merge_stop_waits_for_cleanup_before_stopped(client, monkeypatch):
    merge_id, release = _blocked_merge(client, monkeypatch)
    response = client.post(f"/merge-jobs/{merge_id}/stop")
    assert response.status_code == 200
    assert response.json()["status"] == "stopping"

    # While the worker is alive the job must not report "stopped".
    for _ in range(20):
        assert client.get(f"/merge-jobs/{merge_id}").json()["status"] == "stopping"
        time.sleep(0.005)

    release.set()
    done = wait_for(client, f"/merge-jobs/{merge_id}", {"stopped", "succeeded", "failed"})
    assert done["status"] == "stopped", done
    assert done["merged_pairs"] is None

    merge_dir = client.app.state.merge_manager.merge_dir / merge_id
    assert not merge_dir.exists(), "temporary files must be cleaned before 'stopped'"
    assert client.get(f"/merge-jobs/{merge_id}/download").status_code == 409

    again = client.post(f"/merge-jobs/{merge_id}/stop")
    assert again.status_code == 200
    assert again.json()["status"] == "stopped"


def test_merge_stop_after_success_rejected(client):
    rng = random.Random(43)
    frag = rand_seq(rng, 120)
    r1 = fastq_record("a/1", frag[:75], HIGH * 75)
    r2 = fastq_record("a/2", revcomp(frag[45:]), HIGH * 75)
    job_id = run_qc_job(client, r1, r2)
    done = run_merge_job(client, job_id)
    assert done["status"] == "succeeded"
    response = client.post(f"/merge-jobs/{done['merge_id']}/stop")
    assert response.status_code == 409
    assert client.get(done["downloads"]["merged"]).status_code == 200


def test_merge_download_before_completion_conflicts(client, monkeypatch):
    merge_id, release = _blocked_merge(client, monkeypatch)
    status = client.get(f"/merge-jobs/{merge_id}").json()
    assert status["status"] == "running"
    assert client.get(f"/merge-jobs/{merge_id}/download").status_code == 409
    release.set()
    done = wait_for(client, f"/merge-jobs/{merge_id}", {"succeeded", "failed"})
    assert done["status"] == "succeeded"
    assert client.get(f"/merge-jobs/{merge_id}/download").status_code == 200


def test_merge_stop_during_processing_of_large_input(client):
    """A genuinely running large merge can be stopped and never succeeds."""
    rng = random.Random(47)
    n = 20000
    parts1, parts2 = [], []
    for i in range(n):
        frag = rand_seq(rng, 100)
        parts1.append(fastq_record(f"r{i}/1", frag[:60], HIGH * 60))
        parts2.append(fastq_record(f"r{i}/2", revcomp(frag[30:]), HIGH * 70))
    job_id = run_qc_job(client, "".join(parts1), "".join(parts2))

    merge_id = submit_merge(client, job_id).json()["merge_id"]
    time.sleep(0.01)
    response = client.post(f"/merge-jobs/{merge_id}/stop")
    assert response.status_code in {200, 409}
    if response.status_code == 200:
        done = wait_for(
            client, f"/merge-jobs/{merge_id}", {"stopped", "succeeded", "failed"}, timeout=30
        )
        if response.json()["status"] != "succeeded":
            assert done["status"] == "stopped", done


# ---------------------------------------------------------------------------
# Streaming behaviour and responsiveness
# ---------------------------------------------------------------------------


def test_merge_engine_does_not_read_whole_file(monkeypatch, tmp_path):
    from app import merge as merge_module
    from app import qc

    calls = {"readline": 0}
    original = qc._GzipLineReader.readline

    def counting(self):
        calls["readline"] += 1
        return original(self)

    monkeypatch.setattr(qc._GzipLineReader, "readline", counting)

    rng = random.Random(53)
    total = 200
    r1_parts, r2_parts = [], []
    for i in range(total):
        frag = rand_seq(rng, 100)
        r1_parts.append(fastq_record(f"r{i}/1", frag[:60], HIGH * 60))
        r2_parts.append(fastq_record(f"r{i}/2", revcomp(frag[30:]), HIGH * 70))
    input1 = tmp_path / "R1.fastq.gz"
    input2 = tmp_path / "R2.fastq.gz"
    input1.write_bytes(gz_bytes("".join(r1_parts)))
    input2.write_bytes(gz_bytes("".join(r2_parts)))
    result = merge_module.run_merge(
        input1,
        input2,
        tmp_path / "merged.part",
        threading.Event(),
    )
    assert result.merged_pairs == total
    # Exactly 4 lines per record per side, plus one EOF-probe read per side.
    assert calls["readline"] == total * 4 * 2 + 2


def test_status_responds_while_large_merge_runs(client):
    """Status queries stay responsive during a streaming merge."""
    rng = random.Random(59)
    n = 30000
    parts1, parts2 = [], []
    for i in range(n):
        frag = rand_seq(rng, 100)
        parts1.append(fastq_record(f"r{i}/1", frag[:60], HIGH * 60))
        parts2.append(fastq_record(f"r{i}/2", revcomp(frag[30:]), HIGH * 70))
    job_id = run_qc_job(client, "".join(parts1), "".join(parts2))
    merge_id = submit_merge(client, job_id).json()["merge_id"]

    latencies = []
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        start = time.monotonic()
        payload = client.get(f"/merge-jobs/{merge_id}").json()
        latencies.append(time.monotonic() - start)
        # The QC job and its downloads must stay responsive too.
        qc_payload = client.get(f"/jobs/{job_id}").json()
        assert qc_payload["status"] == "succeeded"
        if payload["status"] in {"succeeded", "failed", "stopped"}:
            break
        time.sleep(0.005)
    assert max(latencies) < 1.0, latencies
    final = wait_for(client, f"/merge-jobs/{merge_id}", {"succeeded", "failed", "stopped"}, 30)
    if final["status"] == "succeeded":
        assert final["merged_pairs"] == n
