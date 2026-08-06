# data_loader.py

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from config import DATASETS, MASTER_SEED, MAX_SAMPLES, TEST_SIZE, OPENML_CACHE_DIR

# Validation metadata, not experiment configuration -- lives here, not in
# config.py. Feature count is an exact expectation (from the paper's own
# tables); row count is a loose sanity band, since minor OpenML-side
# corrections since publication are plausible and shouldn't trip a false
# failure.
#
# Looked up by direct indexing (EXPECTED_SHAPES[name], not .get(name)) so
# a dataset added to config.DATASETS without a corresponding entry here
# fails loudly with a KeyError on first load, rather than silently
# skipping validation for it. validate_dataset_registry() below is the
# complementary check that catches this same class of mistake up front,
# for every dataset at once, before any dataset is processed.
EXPECTED_SHAPES = {
    "bank_marketing": (10578, 7),
    "california": (20640, 8),
    "magic_telescope": (13376, 10),
}


def validate_dataset_registry():
    """
    Startup consistency check between config.DATASETS and this module's
    EXPECTED_SHAPES. Call once, before any dataset is processed -- not
    per-dataset, unlike _validate_fetch() below, which checks an
    individual fetch's actual shape against its own expectation.

    config.DATASETS and EXPECTED_SHAPES are two separate dicts, in two
    separate files, that have to be kept in sync by hand. Without this
    check, a mismatch between them would only surface as a bare KeyError
    the moment that specific dataset happens to be loaded -- possibly
    well into a long run, and with no explanation of what's actually
    wrong. This catches it immediately, for the whole registry at once,
    with a message that names the problem.
    """
    config_names = set(DATASETS.keys())
    metadata_names = set(EXPECTED_SHAPES.keys())

    missing = sorted(config_names - metadata_names)
    extra = sorted(metadata_names - config_names)

    if missing or extra:
        parts = []
        if missing:
            parts.append(
                f"missing from EXPECTED_SHAPES (present in config.DATASETS): {missing}"
            )
        if extra:
            parts.append(
                f"extra in EXPECTED_SHAPES (not present in config.DATASETS): {extra}"
            )
        raise ValueError(
            "Dataset registry mismatch between config.DATASETS and "
            "data_loader.EXPECTED_SHAPES: " + "; ".join(parts) +
            ". Every dataset must have exactly one entry in both."
        )


def _validate_fetch(name, bunch):
    """
    Fail-fast identity check: feature count must match exactly, row count
    must be in the right neighborhood. Deliberately minimal -- fetching by
    a pinned data_id already rules out the name/task-ID resolution bug
    that caused the original incident; this just guards against gross
    data corruption or a wrong ID.
    """
    exp_rows, exp_cols = EXPECTED_SHAPES[name]
    n_rows, n_cols = bunch.data.shape

    if n_cols != exp_cols:
        raise ValueError(
            f"Identity check failed for '{name}': expected {exp_cols} "
            f"features, got {n_cols}. Refusing to proceed with a dataset "
            f"that doesn't match what was requested."
        )
    if not (0.5 * exp_rows <= n_rows <= 1.5 * exp_rows):
        raise ValueError(
            f"Identity check failed for '{name}': expected roughly "
            f"{exp_rows} rows, got {n_rows}. Refusing to proceed with a "
            f"dataset that doesn't match what was requested."
        )


def load_dataset(name):
    ds_config = DATASETS[name]
    print(f"Loading {name} (OpenML id={ds_config['openml_id']})...")

    # target_column intentionally omitted: confirmed via spike that
    # target_column="default-target" is identical to fetch_openml's own
    # default, so stating it explicitly added a parameter with no
    # behavioral effect.
    bunch = fetch_openml(
        data_id=ds_config["openml_id"],
        as_frame=True,
        parser="pandas",
        data_home=OPENML_CACHE_DIR,
        cache=True,
    )
    _validate_fetch(name, bunch)

    X = bunch.data.copy()
    y = bunch.target.copy()

    # Encode classification target to integers. y arrives as category
    # dtype (confirmed via spike, e.g. bank_marketing's '1'/'2' labels);
    # .astype(str) handles that correctly, same as it always has.
    if ds_config["task"] == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=y.name)
    else:
        y = pd.to_numeric(y, errors="coerce")

    # Separate numeric and categorical input features. Includes "category"
    # alongside "object" -- fetch_openml's pandas parser infers proper
    # categorical dtype for nominal features.
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()
    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with any missing values (paper's methodology)
    mask = X[num_cols].notna().all(axis=1) & y.notna()
    if len(cat_cols) > 0:
        mask &= X[cat_cols].notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Cap at MAX_SAMPLES (paper's medium-sized regime)
    if len(X) > MAX_SAMPLES:
        idx = np.random.RandomState(MASTER_SEED).choice(len(X), MAX_SAMPLES, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    # Stratify classification splits to preserve class balance
    stratify = y if ds_config["task"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=MASTER_SEED, stratify=stratify
    )

    # Handle categorical features with OneHotEncoding for tree models
    # Trees can handle OHE features natively and fairly
    if len(cat_cols) > 0:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ohe_train = pd.DataFrame(
            ohe.fit_transform(X_train[cat_cols]),
            columns=ohe.get_feature_names_out(cat_cols)
        )
        ohe_test = pd.DataFrame(
            ohe.transform(X_test[cat_cols]),
            columns=ohe.get_feature_names_out(cat_cols)
        )
        X_train = pd.concat(
            [X_train[num_cols].reset_index(drop=True), ohe_train], axis=1
        )
        X_test = pd.concat(
            [X_test[num_cols].reset_index(drop=True), ohe_test], axis=1
        )
    else:
        X_train = X_train[num_cols].reset_index(drop=True)
        X_test = X_test[num_cols].reset_index(drop=True)

    # Gaussianize features for MLP only — fit on train, apply to test
    qt = QuantileTransformer(output_distribution="normal", random_state=MASTER_SEED)
    X_train_nn = pd.DataFrame(
        qt.fit_transform(X_train), columns=X_train.columns
    )
    X_test_nn = pd.DataFrame(
        qt.transform(X_test), columns=X_test.columns
    )

    print(f"  shape: {X.shape}, task: {ds_config['task']}, "
          f"cat_features: {len(cat_cols)}, num_features: {len(num_cols)}")

    return {
        "name": name,
        "task": ds_config["task"],
        "X_train": X_train,
        "X_test": X_test,
        "X_train_nn": X_train_nn,
        "X_test_nn": X_test_nn,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    validate_dataset_registry()
    for name in DATASETS:
        ds = load_dataset(name)
        print(f"  {ds['name']} loaded successfully\n")