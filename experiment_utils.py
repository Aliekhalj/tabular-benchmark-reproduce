# experiment_utils.py

"""
Small shared infrastructure for the experiment scripts (benchmark.py,
finding2.py, finding3.py): per-dataset lifecycle logging, incremental
CSV writing, timing, and success/failure tracking with a final summary
that distinguishes computation failures from result-persistence failures.

Contains no experiment logic, no metrics, no model code, and no
preprocessing -- purely the failure-isolation, timing, and reporting
scaffolding around the per-dataset loop that's identical across all
three scripts.
"""

import time
import pandas as pd

STAGE_COMPUTATION = "computation"
STAGE_WRITE = "write"


def log_stage(dataset_name, message):
    """Per-dataset lifecycle line, e.g. '[bank_marketing] Loading...'."""
    print(f"[{dataset_name}] {message}")


def format_exception(exc):
    """Consistent 'ExceptionType: message' formatting, used both in the
    live per-dataset failure log and (via log_failed) nowhere else --
    the run summary intentionally omits exception text, since it was
    already printed in full at the moment of failure."""
    return f"{type(exc).__name__}: {exc}"


def log_finished(name, start_time):
    """
    Logs '[name] Finished. (X.XX s)' for a dataset that completed
    successfully (computation and, if applicable, write). start_time is
    a time.perf_counter() value captured by the caller when the
    dataset's processing began. Returns the elapsed time, in case a
    caller wants it; unused by the current scripts.
    """
    elapsed = time.perf_counter() - start_time
    log_stage(name, f"Finished. ({elapsed:.2f} s)")
    return elapsed


def log_failed(name, stage, start_time, exc):
    """
    Logs the two-line '[name] Failed (stage) after X.XX s:' + exception
    message for a dataset that failed at the given stage (STAGE_COMPUTATION
    or STAGE_WRITE). Elapsed is computed the same way as log_finished().
    Returns the elapsed time, same reasoning as log_finished().
    """
    elapsed = time.perf_counter() - start_time
    log_stage(name, f"Failed ({stage}) after {elapsed:.2f} s:")
    print(format_exception(exc))
    return elapsed


class IncrementalCSVWriter:
    """
    Accumulates result rows in memory and rewrites the target CSV in full
    after each successful dataset. Rewriting the whole file (rather than
    appending) means the file on disk is always either the previous
    complete state or the new complete state -- never a partial row --
    which makes duplicate rows structurally impossible.

    If a write fails, the rows just added are rolled back out of the
    in-memory accumulator before the exception is re-raised, so a
    "failed (write)" classification stays a permanently accurate
    statement about the final file rather than a snapshot that could be
    contradicted by a later, unrelated successful write.

    Known, accepted limitation: a process killed during the brief window
    of the to_csv() call itself could still leave a truncated file --
    out of scope, a deliberate choice rather than an oversight.

    Each run starts fresh: the first successful add_rows() call writes
    (overwrites) the target file. There is no cross-run resume.
    """
    def __init__(self, path):
        self.path = path
        self.rows = []

    def add_rows(self, rows):
        n_before = len(self.rows)
        self.rows.extend(rows)
        try:
            pd.DataFrame(self.rows).to_csv(self.path, index=False)
        except Exception:
            self.rows = self.rows[:n_before]  # roll back, see class docstring
            raise


class ExperimentTracker:
    """Tracks which datasets succeeded/failed across a run, distinguishing
    computation failures from write failures, and prints a final summary
    in the same format across all three experiment scripts."""

    def __init__(self):
        self.successful = []
        self.failed = []  # list of (dataset_name, exception, stage) tuples

    def record_success(self, name):
        self.successful.append(name)

    def record_failure(self, name, exc, stage):
        self.failed.append((name, exc, stage))

    def print_summary(self, total_runtime):
        n_success = len(self.successful)
        n_failed = len(self.failed)
        n_total = n_success + n_failed

        print("\n" + "=" * 50)
        print("Run summary")
        print("=" * 50)
        print(f"Datasets processed: {n_total}")
        print(f"Successful: {n_success}")
        print(f"Failed: {n_failed}")
        print(f"Total runtime: {total_runtime:.2f} s")
        print("Successful datasets:")
        for name in self.successful:
            print(f" - {name}")
        print("Failed datasets:")
        for name, exc, stage in self.failed:
            print(f" - {name} [{stage}]")
        print("=" * 50)