"""Dataset loading and preprocessing for CICIDS-2017 / UNSW-NB15.

Provides an identical, deterministic preprocessing pipeline shared by every
experiment (Section 4.1 of the paper): load -> clean -> encode -> scale ->
stratified train/val/test split with a fixed seed.

Raw datasets are **not** committed to the repository. If the configured raw
directory is missing or empty, loading raises a clear ``DataNotFoundError`` that
points the user to ``data/README.md``.

For development and unit testing without the real data, use
:func:`make_synthetic_dataset`, which fabricates *structurally* plausible flow
features (NOT real traffic and NOT experimental results) so the rest of the
pipeline can be exercised.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# pandas / sklearn are imported lazily inside functions so that importing this
# module never hard-fails in a minimal (numpy-only) environment.


class DataNotFoundError(FileNotFoundError):
    """Raised when the configured raw dataset cannot be located."""


@dataclass
class Dataset:
    """A fully preprocessed, split dataset with metadata."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    # The fitted scaler (or None) so adversarial samples can be inverse-scaled
    # for reporting if needed.
    scaler: object = None

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _resolve_paths(cfg: dict) -> List[str]:
    ds = cfg["dataset"]
    raw_dir = ds["raw_dir"]
    if not os.path.isdir(raw_dir):
        raise DataNotFoundError(
            f"Raw data directory '{raw_dir}' not found. Raw datasets are not "
            f"committed to this repo. See data/README.md for how to obtain "
            f"{ds['name']} and where to place the files."
        )
    files = sorted(glob.glob(os.path.join(raw_dir, ds.get("file_glob", "*.csv"))))
    if not files:
        raise DataNotFoundError(
            f"No files matching '{ds.get('file_glob', '*.csv')}' in '{raw_dir}'. "
            f"See data/README.md for the expected file layout."
        )
    return files


def _normalize_columns(columns) -> List[str]:
    """Strip and collapse whitespace in headers so config matching is robust."""
    return [str(c).strip() for c in columns]


def load_dataset(cfg: dict) -> Dataset:
    """Load and preprocess the dataset described by ``cfg``.

    Steps (identical across all experiments):
        1. concatenate raw CSVs;
        2. drop configured identity/leakage columns;
        3. replace +/-inf with NaN and impute (train statistics only);
        4. binarize the label (malicious=1, benign=0);
        5. stratified train/val/test split with the fixed seed;
        6. fit the scaler on the training split only, transform all splits.

    Raises ``DataNotFoundError`` if the raw data is missing.
    """
    import pandas as pd  # local import (heavy dep)

    files = _resolve_paths(cfg)
    ds_cfg = cfg["dataset"]
    pre = cfg.get("preprocessing", {})
    seed = int(cfg.get("seed", 42))
    max_rows = ds_cfg.get("max_rows", None)

    frames = []
    remaining = max_rows
    for f in files:
        df = pd.read_csv(f, nrows=remaining) if remaining else pd.read_csv(f)
        frames.append(df)
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break
    df = pd.concat(frames, ignore_index=True)
    df.columns = _normalize_columns(df.columns)

    # Locate the label column case-insensitively.
    label_col = ds_cfg["label_column"].strip()
    if label_col not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        if label_col.lower() in lower_map:
            label_col = lower_map[label_col.lower()]
        else:
            raise KeyError(
                f"Label column '{ds_cfg['label_column']}' not found. Available "
                f"columns: {list(df.columns)[:20]}... See data/README.md."
            )

    # Drop configured identity/leakage columns (case-insensitive).
    drop_cfg = [c.lower() for c in pre.get("drop_columns", [])]
    to_drop = [c for c in df.columns if c.lower() in drop_cfg and c != label_col]
    df = df.drop(columns=to_drop, errors="ignore")

    # Binarize labels.
    benign = set(str(b).lower() for b in ds_cfg.get("benign_labels", ["benign"]))
    y = (~df[label_col].astype(str).str.lower().isin(benign)).astype(int).to_numpy()
    df = df.drop(columns=[label_col])

    # Keep numeric features only; non-numeric categoricals are one-hot encoded.
    df = _encode_features(df)

    feature_names = list(df.columns)
    x = df.to_numpy(dtype=float)

    # Replace inf with nan, then impute using train stats after the split.
    x[~np.isfinite(x)] = np.nan

    x_train, y_train, x_val, y_val, x_test, y_test = _split(
        x, y, val_size=ds_cfg["val_size"], test_size=ds_cfg["test_size"], seed=seed
    )

    x_train, x_val, x_test = _impute(
        x_train, x_val, x_test, strategy=pre.get("impute", "median")
    )
    x_train, x_val, x_test, scaler = _scale(
        x_train, x_val, x_test, kind=pre.get("scaler", "standard")
    )

    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )


def _encode_features(df):
    import pandas as pd

    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        # One-hot encode low-cardinality categoricals; drop very high-cardinality
        # ones (likely identifiers that slipped past the drop list).
        keep = [c for c in non_numeric if df[c].nunique(dropna=True) <= 32]
        drop = [c for c in non_numeric if c not in keep]
        if drop:
            df = df.drop(columns=drop)
        if keep:
            df = pd.get_dummies(df, columns=keep, dummy_na=False)
    return df


def _split(x, y, val_size, test_size, seed):
    from sklearn.model_selection import train_test_split

    # First split off test, then val from the remainder, both stratified.
    x_tmp, x_test, y_tmp, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_tmp, y_tmp, test_size=rel_val, random_state=seed, stratify=y_tmp
    )
    return x_train, y_train, x_val, y_val, x_test, y_test


def _impute(x_train, x_val, x_test, strategy):
    if strategy == "zero":
        fill = np.zeros(x_train.shape[1])
    elif strategy == "mean":
        fill = np.nanmean(x_train, axis=0)
    else:  # median (default)
        fill = np.nanmedian(x_train, axis=0)
    fill = np.nan_to_num(fill, nan=0.0)

    def apply(a):
        a = a.copy()
        idx = np.where(~np.isfinite(a))
        a[idx] = np.take(fill, idx[1])
        return a

    return apply(x_train), apply(x_val), apply(x_test)


def _scale(x_train, x_val, x_test, kind):
    if kind == "none":
        return x_train, x_val, x_test, None
    if kind == "minmax":
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
    else:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return x_train, x_val, x_test, scaler


def make_synthetic_dataset(
    n_samples: int = 2000,
    n_features: int = 12,
    seed: int = 42,
) -> Dataset:
    """Fabricate a *structurally* plausible dataset for pipeline smoke-testing.

    This is NOT real network traffic and produces NO experimental results; it
    only lets the model/attack/defense code run end-to-end offline. Two
    Gaussian blobs stand in for benign vs. malicious classes.
    """
    rng = np.random.default_rng(seed)
    n_mal = n_samples // 2
    n_ben = n_samples - n_mal
    benign = rng.normal(loc=0.0, scale=1.0, size=(n_ben, n_features))
    malicious = rng.normal(loc=1.5, scale=1.0, size=(n_mal, n_features))
    x = np.vstack([benign, malicious]).astype(float)
    y = np.concatenate([np.zeros(n_ben, dtype=int), np.ones(n_mal, dtype=int)])
    perm = rng.permutation(n_samples)
    x, y = x[perm], y[perm]
    feature_names = [f"f{i}" for i in range(n_features)]

    x_train, y_train, x_val, y_val, x_test, y_test = _split(
        x, y, val_size=0.15, test_size=0.15, seed=seed
    )
    return Dataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=None,
    )
