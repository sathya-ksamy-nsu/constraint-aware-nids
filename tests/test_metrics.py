"""Unit tests for src/metrics.py.

These tests use only NumPy + tiny synthetic arrays -- they do NOT require the
real datasets, PyTorch, or ART. Run with:  pytest tests/test_metrics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics as M  # noqa: E402


def test_clean_accuracy_basic():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])
    assert M.clean_accuracy(y_true, y_pred) == 0.75


def test_clean_accuracy_empty():
    assert M.clean_accuracy(np.array([]), np.array([])) == 0.0


def test_asr_all_evaded():
    # Every malicious sample predicted benign on adv -> ASR == 1.
    y_true = np.array([1, 1, 1])
    y_pred_adv = np.array([0, 0, 0])
    assert M.attack_success_rate(y_true, y_pred_adv) == 1.0


def test_asr_none_evaded():
    y_true = np.array([1, 1, 1])
    y_pred_adv = np.array([1, 1, 1])
    assert M.attack_success_rate(y_true, y_pred_adv) == 0.0


def test_asr_ignores_benign_samples():
    # Benign samples (label 0) must not affect ASR.
    y_true = np.array([1, 0, 1, 0])
    y_pred_adv = np.array([0, 0, 1, 1])  # one of two malicious evaded
    assert M.attack_success_rate(y_true, y_pred_adv) == 0.5


def test_asr_with_clean_filter_denominator():
    # Only malicious samples originally *caught* count toward the denominator.
    y_true = np.array([1, 1, 1, 1])
    y_pred_clean = np.array([1, 1, 0, 0])  # model only caught first two
    y_pred_adv = np.array([0, 1, 0, 0])  # first evaded, second still caught
    # Eligible = first two. Evaded among eligible = 1 -> ASR = 0.5.
    assert M.attack_success_rate(y_true, y_pred_adv, y_pred_clean) == 0.5


def test_asr_no_eligible_returns_zero():
    y_true = np.array([0, 0])  # no malicious samples
    y_pred_adv = np.array([0, 0])
    assert M.attack_success_rate(y_true, y_pred_adv) == 0.0


def test_perturbation_sizes_known_values():
    x_clean = np.array([[0.0, 0.0], [0.0, 0.0]])
    x_adv = np.array([[3.0, 4.0], [0.0, 0.0]])  # row0 L2=5, Linf=4; row1 zeros
    sizes = M.perturbation_sizes(x_clean, x_adv)
    assert np.isclose(sizes["l2_mean"], 2.5)  # (5 + 0) / 2
    assert np.isclose(sizes["linf_mean"], 2.0)  # (4 + 0) / 2
    assert np.isclose(sizes["l2_median"], 2.5)


def test_perturbation_sizes_with_mask():
    x_clean = np.zeros((2, 2))
    x_adv = np.array([[3.0, 4.0], [10.0, 0.0]])
    mask = np.array([True, False])  # only first row counts
    sizes = M.perturbation_sizes(x_clean, x_adv, mask=mask)
    assert np.isclose(sizes["l2_mean"], 5.0)
    assert np.isclose(sizes["linf_mean"], 4.0)


def test_perturbation_sizes_shape_mismatch_raises():
    try:
        M.perturbation_sizes(np.zeros((2, 2)), np.zeros((2, 3)))
    except ValueError:
        return
    raise AssertionError("Expected ValueError on shape mismatch")


def test_valid_sample_rate():
    flags = np.array([True, False, True, True])
    assert M.valid_sample_rate(flags) == 0.75
    assert M.valid_sample_rate(np.array([])) == 0.0


def test_summarize_keys_and_values():
    y_true = np.array([1, 1, 0])
    y_pred_clean = np.array([1, 1, 0])
    y_pred_adv = np.array([0, 1, 0])  # one malicious evaded of two -> ASR 0.5
    x_clean = np.zeros((3, 2))
    x_adv = np.array([[3.0, 4.0], [0.0, 0.0], [9.0, 9.0]])
    validity = np.array([True, True, False])
    out = M.summarize(y_true, y_pred_clean, y_pred_adv, x_clean, x_adv, validity)
    assert np.isclose(out["asr"], 0.5)
    assert out["n_malicious"] == 2
    # Perturbation sizes computed over malicious rows (rows 0 and 1 only).
    assert np.isclose(out["l2_mean"], 2.5)
    assert "valid_sample_rate" in out
    assert np.isclose(out["clean_accuracy"], 1.0)
