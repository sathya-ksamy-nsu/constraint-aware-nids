"""Experiment orchestration (Section 4.6).

Runs the cross-product matrix:

    {each enabled attack} x {unconstrained, constrained} x {no-defense, adv-trained}

for the selected target model(s) and writes per-cell metrics to ``results/`` as
JSON and/or CSV. This module fabricates **no numbers** -- every value written is
computed from an actual attack run on the loaded (real or synthetic) data.

Because attacks/models require torch + ART, the heavy imports live inside the
functions; :mod:`src.metrics` and :mod:`src.constraints` stay import-light.
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import constraints as C
from . import metrics as M


def build_mask(feature_names, cfg) -> C.ConstraintMask:
    """Build the constraint mask for the given feature layout."""
    dataset = cfg.get("dataset", {}).get("name", "cicids2017")
    spec = C.build_default_spec(feature_names, dataset=dataset)
    return C.ConstraintMask(spec)


def _clone_cfg_with_constraints(cfg: dict, enabled: bool) -> dict:
    new = copy.deepcopy(cfg)
    new.setdefault("constraints", {})["enabled"] = enabled
    return new


def evaluate_cell(
    attack_name: str,
    model,
    data,
    cfg: dict,
    mask: C.ConstraintMask,
    constrained: bool,
    defense_label: str,
) -> Dict:
    """Run one matrix cell and return a metrics record (no fabricated numbers)."""
    from . import attacks as A

    cell_cfg = _clone_cfg_with_constraints(cfg, enabled=constrained)

    y_pred_clean_test = model.predict(data.x_test)

    t0 = time.time()
    result = A.run_attack(
        attack_name, model, data.x_test, data.y_test, cell_cfg, mask=mask
    )
    elapsed = time.time() - t0

    # Predictions on adversarial (malicious-only) samples.
    y_pred_adv = model.predict(result.x_adv)
    # Clean predictions restricted to the malicious subset the attack targeted.
    y_pred_clean_mal = model.predict(result.x_clean)

    summary = M.summarize(
        y_true=result.y_true,
        y_pred_clean=y_pred_clean_mal,
        y_pred_adv=y_pred_adv,
        x_clean=result.x_clean,
        x_adv=result.x_adv,
        validity_flags=result.validity_flags,
    )
    record = {
        "attack": attack_name,
        "constrained": constrained,
        "defense": defense_label,
        "clean_accuracy_full_test": M.clean_accuracy(data.y_test, y_pred_clean_test),
        "runtime_sec": round(elapsed, 3),
    }
    record.update(summary)
    return record


def run_matrix(
    model_name: str,
    data,
    cfg: dict,
    attack_names: Optional[List[str]] = None,
    include_defense: bool = True,
) -> List[Dict]:
    """Run the full matrix for one target model and return metric records."""
    from .models import build_model
    from .defense import adversarial_train_mlp

    if attack_names is None:
        attack_names = cfg.get("attacks", {}).get("enabled", ["pgd"])

    mask = build_mask(data.feature_names, cfg)

    # --- undefended model ---
    model = build_model(model_name, data.n_features, cfg)
    model.fit(data.x_train, data.y_train)

    records: List[Dict] = []
    for attack in attack_names:
        for constrained in (False, True):
            records.append(
                evaluate_cell(
                    attack, model, data, cfg, mask, constrained, defense_label="none"
                )
            )

    # --- adversarially trained model (MLP only) ---
    adv_cfg = cfg.get("defense", {}).get("adversarial_training", {})
    if include_defense and adv_cfg.get("enabled", False) and model_name == "mlp":
        defended = build_model(model_name, data.n_features, cfg)
        defended.fit(data.x_train, data.y_train)
        defended = adversarial_train_mlp(
            defended, data.x_train, data.y_train, cfg, mask=mask
        )
        for attack in attack_names:
            for constrained in (False, True):
                records.append(
                    evaluate_cell(
                        attack, defended, data, cfg, mask, constrained,
                        defense_label="adv_training",
                    )
                )

    return records


def write_results(records: List[Dict], cfg: dict, tag: str = "run") -> Dict[str, str]:
    """Persist records to results/ as JSON and/or CSV. Returns written paths."""
    out_dir = cfg.get("output", {}).get("results_dir", "results")
    os.makedirs(out_dir, exist_ok=True)
    formats = cfg.get("output", {}).get("formats", ["json", "csv"])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{tag}_{stamp}"
    written = {}

    if "json" in formats:
        path = os.path.join(out_dir, base + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        written["json"] = path

    if "csv" in formats and records:
        import csv

        path = os.path.join(out_dir, base + ".csv")
        keys = sorted({k for r in records for k in r.keys()})
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        written["csv"] = path

    return written
