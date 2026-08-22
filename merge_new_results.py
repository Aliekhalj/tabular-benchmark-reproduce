# merge_new_results.py -- new file

"""
Explicit, auditable merge of the original 13-dataset results with the
new 10-dataset results into the final 23-dataset artifacts. Reads both
halves fresh and asserts no duplicate keys before writing, rather than
assuming the two files are disjoint. Touches neither source file --
writes new, distinctly-named output for review before promotion.
"""

import pandas as pd


def merge(old_path, new_path, out_path, key_cols):
    old = pd.read_csv(old_path)
    new = pd.read_csv(new_path)

    overlap = set(map(tuple, old[key_cols].values)) & set(map(tuple, new[key_cols].values))
    if overlap:
        raise ValueError(f"Duplicate {key_cols} between {old_path} and {new_path}: {overlap}")

    combined = pd.concat([old, new], ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"{out_path}: {len(old)} + {len(new)} = {len(combined)} rows")


if __name__ == "__main__":
    merge("tuning_results.csv", "tuning_results_new.csv",
          "tuning_results_merged.csv", key_cols=["dataset", "model"])
    merge("cv_results.csv", "cv_results_new.csv",
          "cv_results_merged.csv", key_cols=["dataset", "model"])
    merge("cv_fold_results.csv", "cv_fold_results_new.csv",
          "cv_fold_results_merged.csv", key_cols=["dataset", "model", "fold_index"])