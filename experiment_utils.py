# experiment_utils.py

"""
Small shared infrastructure for the experiment scripts (benchmark.py,
finding2.py, finding3.py): per-dataset lifecycle logging, incremental
CSV writing, and success/failure tracking with a final summary that
distinguishes computation failures from result-persistence failures.

Contains no experiment logic, no metrics, no model code, and no
preprocessing -- purely the failure-isolation and reporting scaffolding
around the per-dataset loop that's identical across all three scripts.
"""

import pandas as pd

STAGE_COMPUTATION = "computation"
STAGE_WRITE = "write"


def log_stage(dataset_name, message):
    """Per-dataset lifecycle line, e.g. '[bank_marketing] Loading...'."""
    print(f"[{dataset_name}] {message}")


def format_exception(exc):
    """Consistent 'ExceptionType: message' formatting, used both in the
    live per-dataset failure log and in the final run summary."""
    return f"{type(exc).__name__}: {exc}"


class IncrementalCSVWriter:
    """
    Accumulates result rows in memory and rewrites the target CSV in full
    after each successful dataset. Rewriting the whole file (rather than
    appending) means the file on disk is always either the previous
    complete state or the new complete state -- never a partial row --
    which makes duplicate rows structurally impossible.

    If a write fails, the rows just added are rolled back out of the
    in-memory accumulator before the exception is re-raised. Without this,
    a failed dataset's rows would remain in memory and could be silently
    included in a later, unrelated successful write -- making a "failed
    (write)" classification in the run summary potentially inaccurate by
    the time the run finishes. The rollback keeps that classification
    permanently trustworthy.

    Known, accepted limitation: a process killed during the brief window
    of the to_csv() call itself could still leave a truncated file. Not
    guarded against here (would need write-to-temp-then-atomic-rename) --
    out of scope for this commit, noted so it's a deliberate choice, not
    an oversight.

    Each run starts fresh: the first successful add_rows() call writes
    (overwrites) the target file, matching the existing to_csv() behavior
    of the scripts being modified here. There is no cross-run resume.
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

    def print_summary(self):
        print("\n" + "=" * 60)
        print("Run summary")
        print("=" * 60)
        print(f"Successful datasets ({len(self.successful)}):")
        for name in self.successful:
            print(f"  - {name}")
        print(f"Failed datasets ({len(self.failed)}):")
        for name, exc, stage in self.failed:
            print(f"  - {name} [{stage}]: {format_exception(exc)}")
        print("=" * 60)