# export_tuned_benchmark.py
"""
Reshapes tuning_results.csv's already-computed, already-evaluated tuned
test scores into a benchmark_results.csv-compatible schema, for
comparison/visualization purposes.

Does NOT refit any model. tuning_results.csv's test_score column was
already produced by tune.py's refit_and_evaluate(), which itself calls
benchmark.py's evaluate() directly -- literally the same function, same
data source, same untouched test set that "benchmark.py using tuned
parameters" would produce if it existed. Re-fitting here would be
redundant compute (hours, for MLP specifically) and a second code path
that could silently drift from tune.py's already leak-checked, already-
verified logic.

Output is a SEPARATE file, benchmark_results_tuned.csv -- does not
overwrite the existing (untuned) benchmark_results.csv, which remains
its own valid, already-verified artifact, needed for the tuned-vs-
untuned comparison.

NOTE on time_s: tuning_results.csv's time_s is the FULL tuning cost
(50-iteration search + final refit), not a single fit time the way
benchmark_results.csv's time_s is. Kept under the same column name for
schema compatibility, but this difference in meaning must be stated
wherever these two files' time_s columns are compared -- they are not
directly comparable numbers.

METHODOLOGICAL NOTE: per the project roadmap, cross-validation (Phase 3)
is intended to sit between tuning and any number treated as "final."
This export is the tuned, SINGLE-SPLIT result -- an interim artifact,
not a replacement for Phase 3's eventual tuned+CV output.
"""

import pandas as pd

tuning = pd.read_csv("tuning_results.csv")

tuned_benchmark = tuning[
    ["dataset", "task", "model", "metric", "test_score", "time_s"]
].rename(columns={"test_score": "score"})

tuned_benchmark.to_csv("benchmark_results_tuned.csv", index=False)

print(f"Wrote {len(tuned_benchmark)} rows to benchmark_results_tuned.csv")
print("NOTE: time_s here is total tuning cost (search+refit), not a single "
      "fit time -- not directly comparable to benchmark_results.csv's time_s "
      "without accounting for this.")