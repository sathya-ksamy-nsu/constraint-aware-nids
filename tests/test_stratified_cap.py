"""Stratified cap helper used by the full-corpus loader."""
import numpy as np

from src.data_loader import _stratified_cap_xy


def test_cap_smaller_than_n_is_noop():
    x = np.zeros((10, 3), dtype=np.float32)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    x2, y2 = _stratified_cap_xy(x, y, n=20, seed=42)
    assert len(y2) == 10


def test_cap_preserves_class_ratio():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(80, dtype=np.int8), np.ones(20, dtype=np.int8)])
    x = rng.normal(size=(100, 2)).astype(np.float32)
    x2, y2 = _stratified_cap_xy(x, y, n=50, seed=42)
    assert len(y2) == 50
    assert abs(float(y2.mean()) - 0.2) < 1e-9
