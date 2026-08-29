"""Evaluation metrics for the constraint-aware adversarial-NIDS study.

All functions here operate on plain NumPy arrays and have **no dependency on
PyTorch or ART**, so they can be unit-tested without the heavy ML stack or the
real datasets.

Metrics implemented:
    * clean_accuracy         - accuracy of a model's predictions on clean data.
    * attack_success_rate    - fraction of originally-malicious samples pushed to
                               a benign prediction (the evasion definition used
                               throughout the paper, Section 4.6).
    * perturbation_sizes     - mean/median L2 and L-inf norms of the perturbation.
    * valid_sample_rate      - fraction of adversarial samples satisfying the
                               constraint mask.

Conventions
-----------
Labels are binary with the convention: 1 == malicious, 0 == benign. A NIDS
evasion is successful when a truly-malicious sample (label 1) is predicted
benign (0) by the target model.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

MALICIOUS = 1
BENIGN = 0


def _as_1d_int(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1).astype(int)


def clean_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Plain classification accuracy on clean inputs."""
    y_true = _as_1d_int(y_true)
    y_pred = _as_1d_int(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def attack_success_rate(
    y_true: np.ndarray,
    y_pred_adv: np.ndarray,
    y_pred_clean: Optional[np.ndarray] = None,
) -> float:
    """Attack success rate (ASR) for NIDS evasion.

    ASR is the fraction of malicious samples that the model predicts as benign
    on the adversarial input. If ``y_pred_clean`` is supplied, only samples that
    were *originally correctly detected as malicious* count toward the
    denominator, which is the stricter and more common definition (an attack
    cannot "succeed" on a sample the model already missed).

    Parameters
    ----------
    y_true : true labels (1 == malicious, 0 == benign).
    y_pred_adv : model predictions on the adversarial inputs.
    y_pred_clean : optional model predictions on the clean inputs.

    Returns
    -------
    float in [0, 1]; returns 0.0 when there are no eligible malicious samples.
    """
    y_true = _as_1d_int(y_true)
    y_pred_adv = _as_1d_int(y_pred_adv)

    malicious_mask = y_true == MALICIOUS
    if y_pred_clean is not None:
        y_pred_clean = _as_1d_int(y_pred_clean)
        # Only count malicious samples the model originally caught.
        eligible = malicious_mask & (y_pred_clean == MALICIOUS)
    else:
        eligible = malicious_mask

    n_eligible = int(np.sum(eligible))
    if n_eligible == 0:
        return 0.0

    evaded = eligible & (y_pred_adv == BENIGN)
    return float(np.sum(evaded) / n_eligible)


def perturbation_sizes(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Mean/median L2 and L-inf perturbation sizes.

    Parameters
    ----------
    x_clean, x_adv : arrays of shape (n_samples, n_features).
    mask : optional boolean array of shape (n_samples,) selecting the rows to
        include (e.g., only successful adversarial samples). When None, all rows
        are used.

    Returns
    -------
    dict with keys l2_mean, l2_median, linf_mean, linf_median.
    """
    x_clean = np.asarray(x_clean, dtype=float)
    x_adv = np.asarray(x_adv, dtype=float)
    if x_clean.shape != x_adv.shape:
        raise ValueError(
            f"x_clean {x_clean.shape} and x_adv {x_adv.shape} must have the same shape"
        )

    delta = (x_adv - x_clean).reshape(x_clean.shape[0], -1)
    if mask is not None:
        mask = np.asarray(mask).reshape(-1).astype(bool)
        delta = delta[mask]

    if delta.shape[0] == 0:
        return {"l2_mean": 0.0, "l2_median": 0.0, "linf_mean": 0.0, "linf_median": 0.0}

    l2 = np.linalg.norm(delta, ord=2, axis=1)
    linf = np.max(np.abs(delta), axis=1)
    return {
        "l2_mean": float(np.mean(l2)),
        "l2_median": float(np.median(l2)),
        "linf_mean": float(np.mean(linf)),
        "linf_median": float(np.median(linf)),
    }


def valid_sample_rate(validity_flags: np.ndarray) -> float:
    """Fraction of adversarial samples that satisfy the constraint mask.

    ``validity_flags`` is a boolean array (True == the sample is a realizable,
    constraint-satisfying perturbation). This is the key metric exposing how
    many "successful" unconstrained evasions are in fact unrealizable.
    """
    flags = np.asarray(validity_flags).reshape(-1).astype(bool)
    if flags.size == 0:
        return 0.0
    return float(np.mean(flags))


def summarize(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_adv: np.ndarray,
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    validity_flags: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Convenience aggregator returning all headline metrics for one run.

    Perturbation sizes are computed over malicious samples only, matching the
    ASR denominator (we only perturb malicious samples in the evasion setting).
    """
    y_true = _as_1d_int(y_true)
    malicious_mask = y_true == MALICIOUS

    result: Dict[str, float] = {
        "clean_accuracy": clean_accuracy(y_true, y_pred_clean),
        "asr": attack_success_rate(y_true, y_pred_adv, y_pred_clean),
        "n_malicious": int(np.sum(malicious_mask)),
    }
    result.update(perturbation_sizes(x_clean, x_adv, mask=malicious_mask))
    if validity_flags is not None:
        result["valid_sample_rate"] = valid_sample_rate(validity_flags)
    return result
