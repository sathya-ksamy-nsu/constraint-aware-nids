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
        1. concatenate raw CSVs (or reload a processed npz / freeze CSV);
        2. drop configured identity/leakage columns;
        3. replace +/-inf with NaN and impute (train statistics only);
        4. binarize the label (malicious=1, benign=0);
        5. stratified train/val/test split with the fixed seed;
        6. fit the scaler on the training split only, transform all splits.

    Raises ``DataNotFoundError`` if the raw data is missing.
    """
    ds_cfg = cfg["dataset"]
    pre = cfg.get("preprocessing", {})
    seed = int(cfg.get("seed", 42))
    rebuild = bool(ds_cfg.get("rebuild_freeze"))
    processed_npz = ds_cfg.get("processed_npz")

    if processed_npz and os.path.isfile(processed_npz) and not rebuild:
        print(f"[data] loading cached preprocessed arrays from {processed_npz}")
        return _load_processed_npz(processed_npz)

    freeze_csv = ds_cfg.get("freeze_csv")
    freeze_n = ds_cfg.get("freeze_n_rows")

    if freeze_csv and os.path.isfile(freeze_csv) and not rebuild:
        import pandas as pd

        encoding = ds_cfg.get("encoding", "latin-1")
        df = pd.read_csv(freeze_csv, encoding=encoding, low_memory=False)
        df.columns = _normalize_columns(df.columns)
        x, y, feature_names = _dataframe_to_xy(df, ds_cfg, pre)
        del df
    else:
        x, y, feature_names = _load_raw_xy(cfg)
        if freeze_n:
            x, y = _stratified_cap_xy(x, y, int(freeze_n), seed)
        if freeze_csv:
            # Optional extract dump only — skipped for full-corpus (freeze_csv null).
            print(f"[data] writing freeze CSV ({len(y)} rows) to {freeze_csv}")
            _write_freeze_csv(x, y, feature_names, freeze_csv)

    x[~np.isfinite(x)] = np.nan

    splits_path = (cfg.get("freeze") or {}).get("splits_path")
    x_train, y_train, x_val, y_val, x_test, y_test = _split_or_freeze(
        x,
        y,
        val_size=ds_cfg["val_size"],
        test_size=ds_cfg["test_size"],
        seed=seed,
        splits_path=splits_path,
    )
    del x, y

    x_train, x_val, x_test = _impute(
        x_train, x_val, x_test, strategy=pre.get("impute", "median")
    )
    x_train, x_val, x_test, scaler = _scale(
        x_train, x_val, x_test, kind=pre.get("scaler", "standard")
    )

    data = Dataset(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
    if processed_npz:
        os.makedirs(os.path.dirname(os.path.abspath(processed_npz)), exist_ok=True)
        _save_processed_npz(processed_npz, data)
        print(f"[data] wrote preprocessed cache {processed_npz}")
    return data


def _load_raw_xy(cfg: dict):
    """Read raw CSVs one file at a time to keep peak memory bounded."""
    import gc
    import pandas as pd

    ds_cfg = cfg["dataset"]
    pre = cfg.get("preprocessing", {})
    encoding = ds_cfg.get("encoding", "latin-1")
    files = _resolve_paths(cfg)
    max_rows = ds_cfg.get("max_rows", None)
    remaining = max_rows
    xs: list = []
    ys: list = []
    feature_names = None
    for path in files:
        nrows = remaining
        print(f"[data] reading {os.path.basename(path)}")
        df_part = pd.read_csv(
            path,
            nrows=nrows,
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",
        )
        df_part.columns = _normalize_columns(df_part.columns)
        x_part, y_part, names = _dataframe_to_xy(df_part, ds_cfg, pre)
        del df_part
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(
                f"Feature schema mismatch in {path}: {names[:5]} vs {feature_names[:5]}"
            )
        xs.append(x_part)
        ys.append(y_part)
        if remaining is not None:
            remaining -= len(y_part)
            if remaining <= 0:
                break
        gc.collect()
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    del xs, ys
    gc.collect()
    print(f"[data] concatenated {len(y)} rows x {x.shape[1]} features")
    return x, y, feature_names


def _dataframe_to_xy(df, ds_cfg: dict, pre: dict):
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

    drop_cfg = [c.lower() for c in pre.get("drop_columns", [])]
    to_drop = [c for c in df.columns if c.lower() in drop_cfg and c != label_col]
    df = df.drop(columns=to_drop, errors="ignore")

    benign = set(str(b).lower() for b in ds_cfg.get("benign_labels", ["benign"]))
    y = (~df[label_col].astype(str).str.lower().isin(benign)).astype(np.int8).to_numpy()
    df = df.drop(columns=[label_col])
    df = _encode_features(df)
    feature_names = list(df.columns)
    x = df.to_numpy(dtype=np.float32)
    return x, y, feature_names


def _write_freeze_csv(x, y, feature_names, path: str):
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(x, columns=feature_names)
    df["Label"] = np.where(y == 0, "BENIGN", "ATTACK")
    df.to_csv(path, index=False)


def _save_processed_npz(path: str, data: Dataset) -> None:
    payload = dict(
        x_train=data.x_train,
        y_train=data.y_train,
        x_val=data.x_val,
        y_val=data.y_val,
        x_test=data.x_test,
        y_test=data.y_test,
        feature_names=np.array(data.feature_names, dtype=object),
    )
    scaler = data.scaler
    if scaler is not None and hasattr(scaler, "mean_"):
        payload["scaler_mean"] = np.asarray(scaler.mean_)
        payload["scaler_scale"] = np.asarray(scaler.scale_)
        payload["scaler_var"] = np.asarray(scaler.var_)
        payload["n_features"] = np.int64(getattr(scaler, "n_features_in_", data.n_features))
        payload["n_samples_seen"] = np.int64(getattr(scaler, "n_samples_seen_", len(data.y_train)))
    np.savez_compressed(path, **payload)


def _load_processed_npz(path: str) -> Dataset:
    z = np.load(path, allow_pickle=True)
    scaler = None
    if "scaler_mean" in z.files:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = z["scaler_mean"]
        scaler.scale_ = z["scaler_scale"]
        scaler.var_ = z["scaler_var"]
        scaler.n_features_in_ = int(z["n_features"])
        scaler.n_samples_seen_ = int(z["n_samples_seen"])
    return Dataset(
        x_train=z["x_train"],
        y_train=z["y_train"],
        x_val=z["x_val"],
        y_val=z["y_val"],
        x_test=z["x_test"],
        y_test=z["y_test"],
        feature_names=list(z["feature_names"]),
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


def _stratified_cap_xy(x, y, n: int, seed: int):
    """Stratified subsample of ``n`` rows from arrays."""
    if len(y) <= n:
        return x, y
    from sklearn.model_selection import train_test_split

    keep_idx, _ = train_test_split(
        np.arange(len(y)),
        train_size=n,
        random_state=seed,
        stratify=y,
    )
    keep_idx = np.sort(keep_idx)
    return x[keep_idx], y[keep_idx]


def _stratified_cap(df, ds_cfg: dict, n: int, seed: int):
    """Stratified subsample of ``n`` rows using the configured label column."""
    if len(df) <= n:
        return df
    label_col = ds_cfg["label_column"].strip()
    if label_col not in df.columns:
        lower_map = {c.lower(): c for c in df.columns}
        label_col = lower_map.get(label_col.lower(), label_col)
    benign = set(str(b).lower() for b in ds_cfg.get("benign_labels", ["benign"]))
    y = (~df[label_col].astype(str).str.lower().isin(benign)).astype(int)
    from sklearn.model_selection import train_test_split

    keep_idx, _ = train_test_split(
        np.arange(len(df)),
        train_size=n,
        random_state=seed,
        stratify=y,
    )
    keep_idx = np.sort(keep_idx)
    return df.iloc[keep_idx].reset_index(drop=True)


def _split_or_freeze(x, y, val_size, test_size, seed, splits_path=None):
    n = len(y)
    if splits_path and os.path.isfile(splits_path):
        z = np.load(splits_path)
        if int(z["n_rows"]) == n:
            tr, va, te = z["train"], z["val"], z["test"]
            return x[tr], y[tr], x[va], y[va], x[te], y[te]
    x_train, y_train, x_val, y_val, x_test, y_test, idx = _split(
        x, y, val_size, test_size, seed, return_indices=True
    )
    if splits_path:
        os.makedirs(os.path.dirname(os.path.abspath(splits_path)), exist_ok=True)
        np.savez_compressed(
            splits_path,
            train=idx["train"],
            val=idx["val"],
            test=idx["test"],
            n_rows=n,
            seed=seed,
        )
    return x_train, y_train, x_val, y_val, x_test, y_test


def _split(x, y, val_size, test_size, seed, return_indices=False):
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    idx_tmp, idx_test = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_tmp, test_size=rel_val, random_state=seed, stratify=y[idx_tmp]
    )
    parts = (
        x[idx_train],
        y[idx_train],
        x[idx_val],
        y[idx_val],
        x[idx_test],
        y[idx_test],
    )
    if return_indices:
        return (*parts, {"train": idx_train, "val": idx_val, "test": idx_test})
    return parts


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
