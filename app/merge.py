"""Streaming consensus merger for QC-passed paired-end reads.

For each kept pair published by a QC job, the R2 read is reverse-complemented
(its quality line reversed) so both ends read the same strand.  A pair is
merged only when the tail of R1 and the head of the reverse-complemented R2
share exactly one candidate overlap that

* is at least ``MIN_OVERLAP`` (20) bases long, and
* has at most ``MAX_MISMATCHES`` (1) mismatching base.

If no length qualifies the pair is counted as ``no_overlap``; if more than one
length qualifies it is counted as ``ambiguous``; pairs containing any base
outside ``ACGT`` are counted as ``non_acgt``.  None of those produce output.

Consensus rules inside the overlap:

* equal bases: keep the base and the higher of the two qualities;
* differing bases: when the Phred difference is at least ``MIN_QUAL_DELTA``
  (5) the higher-quality end wins (its base and its quality); otherwise the
  base becomes ``N`` with the lowest quality ``!``.

Outside the overlap the unmerged flanks are copied through unchanged.  Merged
records are written as single-end FASTQ in the original pair order, named by
the shared identifier with the ``/1`` suffix removed.

Like the QC engine, everything is streamed record by record; the output is
written to a ``.part`` file and only renamed into place by the caller after a
successful return, so a corrupt input gzip, a write failure, or a stop request
never publishes a half-written result.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# The merge engine reuses the QC module's streaming gzip reader/writer and
# record validation so both engines share exactly the same strictness
# (gzip corruption detection, four-line records, Phred+33 range checks).
from .qc import (
    JobStopped,
    QCError,
    _check_record,
    _GzipLineReader,
    _GzipWriter,
    _strip_eol,
)

MIN_OVERLAP = 20
MAX_MISMATCHES = 1
MIN_QUAL_DELTA = 5

_N_BYTE = ord("N")
_LOWEST_QUAL_BYTE = ord("!")  # Phred 0

_ACGT_BYTES = frozenset(b"ACGT")
_RC_TABLE = bytes.maketrans(b"ACGT", b"TGCA")

PROGRESS_EVERY_PAIRS = 256

# Classification of one pair.
MERGED = "merged"
NO_OVERLAP = "no_overlap"
AMBIGUOUS = "ambiguous"
NON_ACGT = "non_acgt"


@dataclass(frozen=True)
class MergeResult:
    total_pairs: int
    merged_pairs: int
    no_overlap_pairs: int
    ambiguous_pairs: int
    non_acgt_pairs: int


ProgressCallback = Callable[[int], None]


def reverse_complement(sequence: bytes) -> bytes:
    """Reverse-complement an ACGT-only sequence."""
    return sequence.translate(_RC_TABLE)[::-1]


def _count_mismatches(a: bytes, b: bytes, limit: int) -> int:
    """Count position-wise mismatches, stopping once *limit* is exceeded."""
    mismatches = 0
    for x, y in zip(a, b):
        if x != y:
            mismatches += 1
            if mismatches > limit:
                return mismatches
    return mismatches


def _classify_overlap(seq1: bytes, seq2rc: bytes) -> tuple[str, int]:
    """Classify the tail(R1)/head(R2rc) overlap candidates.

    Returns ``(kind, length)`` where *kind* is one of ``NO_OVERLAP``,
    ``AMBIGUOUS`` or ``MERGED``; *length* is the unique overlap length and is
    only meaningful for ``MERGED``.
    """
    candidate = -1
    longest = min(len(seq1), len(seq2rc))
    for length in range(longest, MIN_OVERLAP - 1, -1):
        tail = seq1[len(seq1) - length :]
        head = seq2rc[:length]
        if _count_mismatches(tail, head, MAX_MISMATCHES) <= MAX_MISMATCHES:
            if candidate != -1:
                return AMBIGUOUS, 0
            candidate = length
    if candidate == -1:
        return NO_OVERLAP, 0
    return MERGED, candidate


def _consensus_overlap(
    bases1: bytes, bases2: bytes, quals1: bytes, quals2: bytes
) -> tuple[bytes, bytes]:
    """Build the consensus sequence/quality for one aligned overlap."""
    out_bases = bytearray(len(bases1))
    out_quals = bytearray(len(bases1))
    for i in range(len(bases1)):
        b1 = bases1[i]
        b2 = bases2[i]
        q1 = quals1[i]
        q2 = quals2[i]
        if b1 == b2:
            out_bases[i] = b1
            out_quals[i] = q1 if q1 >= q2 else q2
        elif q1 - q2 >= MIN_QUAL_DELTA:
            out_bases[i] = b1
            out_quals[i] = q1
        elif q2 - q1 >= MIN_QUAL_DELTA:
            out_bases[i] = b2
            out_quals[i] = q2
        else:
            out_bases[i] = _N_BYTE
            out_quals[i] = _LOWEST_QUAL_BYTE
    return bytes(out_bases), bytes(out_quals)


def merge_pair(
    seq1: bytes, qual1: bytes, seq2: bytes, qual2: bytes
) -> tuple[str, bytes, bytes]:
    """Classify one pair and, when possible, build the consensus read.

    ``seq2``/``qual2`` are the raw R2 read; they are reverse-complemented /
    reversed here.  Returns ``(kind, sequence, quality)``; the sequence and
    quality are empty unless *kind* is ``MERGED``.
    """
    if not (set(seq1) <= _ACGT_BYTES and set(seq2) <= _ACGT_BYTES):
        return NON_ACGT, b"", b""

    seq2rc = reverse_complement(seq2)
    qual2rc = qual2[::-1]

    kind, length = _classify_overlap(seq1, seq2rc)
    if kind != MERGED:
        return kind, b"", b""

    head_bases = seq1[: len(seq1) - length]
    head_quals = qual1[: len(seq1) - length]
    mid_bases, mid_quals = _consensus_overlap(
        seq1[len(seq1) - length :],
        seq2rc[:length],
        qual1[len(seq1) - length :],
        qual2rc[:length],
    )
    tail_bases = seq2rc[length:]
    tail_quals = qual2rc[length:]
    return (
        MERGED,
        head_bases + mid_bases + tail_bases,
        head_quals + mid_quals + tail_quals,
    )


def run_merge(
    r1_path: Path,
    r2_path: Path,
    output_part: Path,
    stop_event: threading.Event,
    *,
    progress_cb: ProgressCallback | None = None,
) -> MergeResult:
    """Merge the published QC outputs into a single-end consensus FASTQ.

    ``output_part`` is a complete, valid gzip file when this function returns;
    the caller renames it only after success, so failures and stops never
    expose a partial result.
    """
    reader1 = reader2 = writer = None
    try:
        reader1 = _GzipLineReader(r1_path, "R1", stop_event)
        reader2 = _GzipLineReader(r2_path, "R2", stop_event)
        writer = _GzipWriter(output_part)

        total = 0
        merged = 0
        no_overlap = 0
        ambiguous = 0
        non_acgt = 0
        while True:
            if stop_event.is_set():
                raise JobStopped()

            header1 = reader1.readline()
            header2 = reader2.readline()
            if not header1 and not header2:
                break
            if not header1 or not header2:
                missing = "R2" if header1 else "R1"
                other = "R1" if missing == "R2" else "R2"
                raise QCError(
                    "pair_count_mismatch",
                    f"{other} has more records than {missing}; "
                    "the two ends contain different numbers of reads",
                )

            total += 1
            lines1 = [
                header1,
                reader1.readline(),
                reader1.readline(),
                reader1.readline(),
            ]
            lines2 = [
                header2,
                reader2.readline(),
                reader2.readline(),
                reader2.readline(),
            ]
            for side, lines in (("R1", lines1), ("R2", lines2)):
                if any(line == b"" for line in lines[1:]):
                    raise QCError(
                        "malformed_record",
                        f"{side}: record {total} has fewer than four lines "
                        "(truncated FASTQ record)",
                    )

            id1, _ = _check_record(lines1, "R1", total, b"/1")
            id2, _ = _check_record(lines2, "R2", total, b"/2")
            if id1 != id2:
                raise QCError(
                    "pair_name_mismatch",
                    f"record {total}: R1 read name "
                    f"{id1.decode('utf-8', 'replace')!r} does not match R2 "
                    f"read name {id2.decode('utf-8', 'replace')!r}",
                )

            kind, out_seq, out_qual = merge_pair(
                _strip_eol(lines1[1]),
                _strip_eol(lines1[3]),
                _strip_eol(lines2[1]),
                _strip_eol(lines2[3]),
            )
            if kind == MERGED:
                merged += 1
                if stop_event.is_set():
                    raise JobStopped()
                writer.write(
                    b"@" + id1 + b"\n" + out_seq + b"\n+\n" + out_qual + b"\n"
                )
            elif kind == NO_OVERLAP:
                no_overlap += 1
            elif kind == AMBIGUOUS:
                ambiguous += 1
            else:
                non_acgt += 1

            if progress_cb is not None and (
                total % PROGRESS_EVERY_PAIRS == 0
            ):
                progress_cb(total)

        if stop_event.is_set():
            raise JobStopped()
        return MergeResult(
            total_pairs=total,
            merged_pairs=merged,
            no_overlap_pairs=no_overlap,
            ambiguous_pairs=ambiguous,
            non_acgt_pairs=non_acgt,
        )
    finally:
        # If the body already raised, keep that error: a secondary close
        # failure must not mask it.
        failed = sys.exc_info()[0] is not None
        for reader in (reader1, reader2):
            if reader is not None:
                reader.close()
        if writer is not None:
            try:
                writer.close()
            except Exception:
                if not failed:
                    raise
