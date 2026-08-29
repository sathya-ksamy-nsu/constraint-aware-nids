"""Unit tests for src/constraints.py (the constraint mask).

Uses only NumPy + tiny synthetic arrays -- no real datasets, PyTorch, or ART.
Run with:  pytest tests/test_constraints.py

Synthetic feature layout (see build_synthetic_spec):
    0 protocol      IMMUTABLE
    1 dst_port      IMMUTABLE
    2 fwd_packets   INCREASE_ONLY, integer, >= 0
    3 bwd_packets   INCREASE_ONLY, integer, >= 0
    4 total_packets derived: == fwd + bwd
    5 mean_iat      FREE, >= 0
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constraints as C  # noqa: E402


def _clean_sample():
    # protocol=6(TCP), dst_port=80, fwd=3, bwd=2, total=5, mean_iat=1.0
    return np.array([[6.0, 80.0, 3.0, 2.0, 5.0, 1.0]])


def make_mask():
    return C.ConstraintMask(C.build_synthetic_spec())


# --------------------------------------------------------------- immutability
def test_immutable_features_reset_on_project():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 0] = 17.0   # tamper protocol (immutable)
    x_adv[0, 1] = 443.0  # tamper dst_port (immutable)
    proj = mask.project(x_adv, x0)
    assert proj[0, 0] == 6.0
    assert proj[0, 1] == 80.0


def test_immutable_change_is_invalid():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 0] = 17.0  # protocol changed
    assert mask.is_valid(x_adv, x0)[0] == False  # noqa: E712


# --------------------------------------------------------------- direction
def test_increase_only_direction_projected():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 2] = 1.0  # fwd_packets decreased 3 -> 1 (not allowed)
    proj = mask.project(x_adv, x0)
    # Decrease clamped back to the clean value (no negative delta allowed).
    assert proj[0, 2] >= x0[0, 2]


def test_increase_only_decrease_is_invalid():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 3] = 0.0  # bwd_packets decreased 2 -> 0
    # total also becomes inconsistent, but the direction violation alone
    # already makes it invalid.
    assert mask.is_valid(x_adv, x0)[0] == False  # noqa: E712


def test_increase_allowed():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 2] = 6.0  # fwd 3 -> 6 (increase, allowed)
    x_adv[0, 4] = 8.0  # keep total consistent (6 + 2)
    assert mask.is_valid(x_adv, x0)[0] == True  # noqa: E712


# --------------------------------------------------------------- ranges
def test_range_non_negativity_enforced():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 5] = -3.0  # mean_iat negative (below lo=0)
    proj = mask.project(x_adv, x0)
    assert proj[0, 5] >= 0.0


def test_out_of_range_is_invalid():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 5] = -0.5  # below lower bound
    assert mask.is_valid(x_adv, x0)[0] == False  # noqa: E712


def test_integer_features_rounded():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 2] = 5.7  # fwd_packets should be integer after projection
    x_adv[0, 3] = 2.0
    proj = mask.project(x_adv, x0)
    assert proj[0, 2] == float(round(5.7))


# --------------------------------------------------------------- interdependency
def test_interdependency_repaired_on_project():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 2] = 10.0  # fwd increased
    x_adv[0, 3] = 4.0   # bwd increased
    x_adv[0, 4] = 5.0   # total left stale (should become 14)
    proj = mask.project(x_adv, x0)
    assert proj[0, 4] == proj[0, 2] + proj[0, 3]


def test_interdependency_violation_is_invalid():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 2] = 10.0
    x_adv[0, 3] = 4.0
    x_adv[0, 4] = 5.0  # total inconsistent with fwd+bwd
    assert mask.is_valid(x_adv, x0)[0] == False  # noqa: E712


def test_fully_consistent_sample_is_valid():
    mask = make_mask()
    x0 = _clean_sample()
    x_adv = x0.copy()
    x_adv[0, 2] = 10.0
    x_adv[0, 3] = 4.0
    x_adv[0, 4] = 14.0  # consistent total
    x_adv[0, 5] = 2.0   # iat increased, still >= 0
    assert mask.is_valid(x_adv, x0)[0] == True  # noqa: E712


def test_project_output_is_valid():
    # Projecting an arbitrary perturbation must yield a constraint-valid sample.
    rng = np.random.default_rng(0)
    mask = make_mask()
    x0 = np.tile(_clean_sample(), (20, 1))
    noise = rng.normal(scale=5.0, size=x0.shape)
    proj = mask.project(x0 + noise, x0)
    flags = mask.is_valid(proj, x0)
    assert flags.all()


def test_valid_sample_rate_matches_flags():
    mask = make_mask()
    x0 = np.tile(_clean_sample(), (4, 1))
    x_adv = x0.copy()
    # Make two of four invalid by tampering the immutable protocol.
    x_adv[0, 0] = 99.0
    x_adv[1, 0] = 99.0
    rate = mask.valid_sample_rate(x_adv, x0)
    assert np.isclose(rate, 0.5)


# --------------------------------------------------------------- spec builders
def test_build_default_spec_marks_protocol_immutable():
    names = ["Protocol", "Dst Port", "Total Fwd Packets", "Flow Duration"]
    spec = C.build_default_spec(names)
    mutable = spec.mutable_mask()
    # "Protocol", "Dst Port" (port), contain immutable keywords.
    assert mutable[0] == False  # noqa: E712  Protocol
    assert mutable[1] == False  # noqa: E712  Dst Port
    assert mutable[2] == True   # noqa: E712  Total Fwd Packets
    assert mutable[3] == True   # noqa: E712  Flow Duration


def test_build_default_spec_wires_sum_relation():
    names = ["Total Fwd Packets", "Total Backward Packets", "Total Packets"]
    spec = C.build_default_spec(names)
    assert len(spec.interdependencies) == 1
    mask = C.ConstraintMask(spec)
    x0 = np.array([[3.0, 2.0, 5.0]])
    x_adv = np.array([[4.0, 2.0, 5.0]])  # total stale (should be 6)
    proj = mask.project(x_adv, x0)
    assert proj[0, 2] == 6.0
