import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

DATASETS = {
    "bank_marketing":  {"file": "bank_marketing.arff",  "task": "classification"},
    "california":      {"file": "california.arff",       "task": "regression"},
    "magic_telescope": {"file": "magic_telescope.arff",  "task": "classification"},
}

def load_dataset(name):
    config = DATASETS[name]
    print(f"Loading {name}...")

    raw, meta = arff.loadarff(config["file"])
    df = pd.DataFrame(raw)

    # Decode byte strings ARFF files produce
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.decode("utf-8")

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Encode classification target to integers
    if config["task"] == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
    else:
        y = pd.to_numeric(y, errors="coerce")

    # Separate numeric and categorical input features
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    X[num_cols] = X[num_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with any missing values (paper's methodology)
    mask = X[num_cols].notna().all(axis=1) & y.notna()
    if len(cat_cols) > 0:
        mask &= X[cat_cols].notna().all(axis=1)
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # Cap at 10,000 samples (paper's medium-sized regime)
    if len(X) > 10000:
        idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    # Stratify classification splits to preserve class balance
    stratify = y if config["task"] == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
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
        X_test  = X_test[num_cols].reset_index(drop=True)

    # Gaussianize features for MLP only — fit on train, apply to test
    # This follows the paper's preprocessing for neural networks
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_nn = pd.DataFrame(
        qt.fit_transform(X_train), columns=X_train.columns
    )
    X_test_nn = pd.DataFrame(
        qt.transform(X_test), columns=X_test.columns
    )

    print(f"  shape: {X.shape}, task: {config['task']}, "
          f"cat_features: {len(cat_cols)}, num_features: {len(num_cols)}")

    return {
        "name":       name,
        "task":       config["task"],
        "X_train":    X_train,      # raw (OHE applied) — for tree models
        "X_test":     X_test,
        "X_train_nn": X_train_nn,   # gaussianized — for MLP
        "X_test_nn":  X_test_nn,
        "y_train":    y_train,
        "y_test":     y_test,
    }

if __name__ == "__main__":
    for name in DATASETS:
        ds = load_dataset(name)
        print(f"  {ds['name']} loaded successfully\n")