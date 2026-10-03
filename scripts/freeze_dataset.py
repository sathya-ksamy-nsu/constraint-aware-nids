"""Build the CICIDS-2017 freeze extract, split indices, and DATA_FREEZE.md.

Run from topic-1-adversarial-nids/:

    python scripts/freeze_dataset.py
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import yaml
    from sklearn.ensemble import RandomForestClassifier

    from src.constraints import build_default_spec
    from src.data_loader import load_dataset
    from src.metrics import clean_accuracy, f1_malicious
    from src.run_meta import write_run_meta

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["dataset"]["raw_dir"]
    files = sorted(
        os.path.join(raw_dir, name)
        for name in os.listdir(raw_dir)
        if name.lower().endswith(".csv")
    )
    if not files:
        print("No CICIDS CSVs in", raw_dir, file=sys.stderr)
        print("See data/README.md or scripts/download_cicids2017.py", file=sys.stderr)
        return 2

    file_rows = []
    for path in files:
        # Count lines cheaply (includes header).
        with open(path, "rb") as fh:
            n_lines = sum(1 for _ in fh)
        file_rows.append(
            {
                "name": os.path.basename(path),
                "bytes": os.path.getsize(path),
                "sha256": _sha256(path),
                "lines_incl_header": n_lines,
            }
        )
        print(f"[hash] {os.path.basename(path)} {file_rows[-1]['sha256'][:12]}…")

    print("[info] loading full corpus / splits (first run concatenates raw CSVs)…")
    data = load_dataset(cfg)
    spec = build_default_spec(data.feature_names, dataset="cicids2017")
    mask_hash = spec.fingerprint()
    n_explicit = sum(
        1
        for name in data.feature_names
        if __import__("src.cicids_mask", fromlist=["cicids2017_feature_spec"]).cicids2017_feature_spec(name)
        is not None
    )

    # Sanity-check a cheap sklearn RF — not a paper result. Cap fit size so the
    # freeze script stays a documentation step, not a 200-tree training run.
    from sklearn.model_selection import train_test_split

    seed = int(cfg.get("seed", 42))
    n_sanity = min(100_000, len(data.y_train))
    if len(data.y_train) > n_sanity:
        s_idx, _ = train_test_split(
            np.arange(len(data.y_train)),
            train_size=n_sanity,
            random_state=seed,
            stratify=data.y_train,
        )
        x_s, y_s = data.x_train[s_idx], data.y_train[s_idx]
    else:
        x_s, y_s = data.x_train, data.y_train
    rf = RandomForestClassifier(
        n_estimators=40, max_depth=20, n_jobs=4, random_state=seed
    )
    rf.fit(x_s, y_s)
    pred = rf.predict(data.x_test)
    sanity_acc = clean_accuracy(data.y_test, pred)
    sanity_f1 = f1_malicious(data.y_test, pred)
    print(f"[sanity] sklearn RF clean acc={sanity_acc:.4f} f1_mal={sanity_f1:.4f}")

    freeze_csv = cfg["dataset"].get("freeze_csv")
    freeze_csv_hash = _sha256(freeze_csv) if freeze_csv and os.path.isfile(freeze_csv) else None
    splits_path = cfg.get("freeze", {}).get("splits_path")
    processed_npz = cfg["dataset"].get("processed_npz")
    scope = (cfg.get("freeze") or {}).get("scope", "unspecified")
    freeze_n = cfg["dataset"].get("freeze_n_rows")

    os.makedirs("results", exist_ok=True)
    md_path = os.path.join("results", "DATA_FREEZE.md")
    now = datetime.now(timezone.utc).isoformat()
    if freeze_n:
        freeze_line = (
            f"- **Freeze extract:** stratified `freeze_n_rows={freeze_n}` with seed "
            f"`{cfg.get('seed')}` written to `{freeze_csv}` (gitignored)."
        )
    else:
        freeze_line = (
            f"- **Freeze extract:** **full corpus** (`freeze_n_rows` unset). "
            f"No processed CSV dump. Preprocessed arrays: `{processed_npz}` (gitignored)."
        )
    lines = [
        "# CICIDS-2017 data freeze",
        "",
        f"- **Frozen at (UTC):** {now}",
        f"- **Scope:** `{scope}`",
        f"- **Source:** official MachineLearningCSV (CICFlowMeter), mirrored as Hugging Face `bencorn/CICIDS2017` `csvs/MachineLearningCSV.zip`.",
        f"- **Citation:** Sharafaldin, Lashkari & Ghorbani (2018), ICISSP. Cite UNB CICIDS-2017; do not redistribute raw CSVs from this repo.",
        f"- **Label column:** `{cfg['dataset']['label_column']}` confirmed. Benign labels: {cfg['dataset']['benign_labels']}.",
        freeze_line,
        f"- **Freeze CSV SHA-256:** `{freeze_csv_hash}`",
        f"- **Splits:** `{splits_path}` (train/val/test indices into the frozen row order; seed {cfg.get('seed')}).",
        f"- **Split sizes:** train={len(data.y_train)} val={len(data.y_val)} test={len(data.y_test)} features={data.n_features}",
        f"- **Class balance (test malicious fraction):** {float(data.y_test.mean()):.4f}",
        f"- **Mask version (SHA-256):** `{mask_hash}`",
        f"- **Explicit CICIDS columns in mask:** {n_explicit} / {data.n_features}",
        f"- **Attack victim cap:** `attacks.max_samples={cfg.get('attacks', {}).get('max_samples')}`; HopSkipJump `{cfg.get('attacks', {}).get('hopskipjump')}`.",
        f"- **Adv-training cap:** `defense.adversarial_training.max_train_samples={cfg.get('defense', {}).get('adversarial_training', {}).get('max_train_samples')}` (undefended models still train on the full train split).",
        f"- **MLP batch_size:** `{cfg.get('models', {}).get('mlp', {}).get('batch_size')}`.",
        f"- **Sanity (not a paper result):** sklearn RF n_estimators=40 max_depth=20 fit on {n_sanity} train rows; clean accuracy={sanity_acc:.4f}, F1-malicious={sanity_f1:.4f} on the freeze test split. Leakage columns Flow ID / IPs / Timestamp are absent from this ML CSV export.",
        "",
        "## Source file hashes",
        "",
        "| File | Bytes | Lines (incl. header) | SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in file_rows:
        lines.append(
            f"| `{row['name']}` | {row['bytes']} | {row['lines_incl_header']} | `{row['sha256']}` |"
        )
    lines.append("")
    lines.append("## Honesty")
    lines.append("")
    if freeze_n:
        honesty = (
            "This freeze is **real CICIDS-2017 traffic**, not synthetic blobs. "
            "It is a stratified subsample of the official MachineLearningCSV tables, "
            "documented so later full-corpus runs can be compared. Do not paste the "
            "sklearn RF sanity numbers into Section 6; fill those cells from "
            "`python experiments/run_experiment.py` (`status=ok`)."
        )
    else:
        honesty = (
            "This freeze is the **full official MachineLearningCSV corpus** of "
            "CICIDS-2017 (every concatenated CSV row after dropping malformed "
            "lines), not a stratified 80k extract and not synthetic blobs. "
            "Attack evaluation still caps victims (`attacks.max_samples`). "
            "Adversarial-training PGD uses `max_train_samples` of the train split; "
            "undefended MLP and RF train on the full train split. Do not paste the "
            "sklearn RF sanity numbers into Section 6; fill those cells from "
            "`python experiments/run_experiment.py` (`status=ok`)."
        )
    lines.append(honesty)
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[info] wrote", md_path)

    write_run_meta(
        os.path.join("results", "run_meta.json"),
        label="environment",
        extra={
            "mask_hash": mask_hash,
            "n_train": int(len(data.y_train)),
            "n_val": int(len(data.y_val)),
            "n_test": int(len(data.y_test)),
            "sanity_rf_clean_accuracy": sanity_acc,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
