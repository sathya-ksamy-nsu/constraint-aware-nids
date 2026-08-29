"""The constraint mask -- the central artifact of the paper (Section 4.4).

This module defines a *realistic-perturbation* specification for NIDS flow
features and the machinery to enforce it. It has **no dependency on PyTorch or
ART** (only NumPy) so it can be unit-tested on synthetic arrays without the real
datasets.

The mask encodes, per feature:
    * mutability      - whether an attacker can change the feature at all;
    * direction       - +1 (may only increase), -1 (may only decrease), 0 (free);
    * bounds [lo, hi] - physically/protocol legal range;
and, across features:
    * interdependencies - equality/inequality relations that must continue to
      hold after perturbation (e.g. total = fwd + bwd; rate = bytes / duration).

Two operations are provided:
    * ``project`` - clip/project a perturbed sample back onto the feasible set so
      attacks optimize *within* the realizable region;
    * ``is_valid`` / ``valid_sample_rate`` - a validity checker used to measure
      how many adversarial samples are actually realizable.

IMPORTANT (assumptions & scope)
-------------------------------
The generic engine below is dataset-agnostic. The *concrete* per-feature entries
(which exact CICIDS-2017 / UNSW-NB15 columns are immutable, their ranges, and
their interdependency formulas) depend on the chosen dataset's exact column
names and are marked TODO in :func:`build_default_spec`. The relationships
encoded there (non-negativity of counts/durations, total = fwd + bwd, immutable
protocol/port/flag-type identifiers, one-directional add-only manipulation) are
the domain assumptions justified in the paper and in ``data/README.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FREE = 0
INCREASE_ONLY = 1
DECREASE_ONLY = -1


@dataclass
class FeatureSpec:
    """Per-feature constraint entry.

    Parameters
    ----------
    name : feature/column name.
    mutable : if False the attacker cannot change this feature at all
        (delta forced to 0). Immutable examples: protocol id, port, flag-type
        indicators, victim/destination-derived counters.
    direction : one of FREE(0), INCREASE_ONLY(+1), DECREASE_ONLY(-1). Many flow
        features are add-only from an attacker's perspective (you can send more
        packets or add delay, but cannot un-send packets already transmitted).
    lo, hi : inclusive lower/upper bound on the *feature value* (not the
        perturbation). Use -inf / +inf for unbounded. Counts and durations are
        non-negative, so lo defaults to 0.0 for mutable numeric features unless
        overridden.
    integer : if True the feature is integer-valued (counts, flag tallies) and
        projected values are rounded.
    """

    name: str
    mutable: bool = True
    direction: int = FREE
    lo: float = 0.0
    hi: float = np.inf
    integer: bool = False


# An interdependency is a callable that repairs a batch X (n, d) in place-safe
# fashion and returns the repaired batch, given a name->index map. Keeping them
# as small functions lets us express both equalities (total = fwd + bwd) and
# clamping inequalities (bytes <= packets * MSS).
Interdependency = Callable[[np.ndarray, Dict[str, int]], np.ndarray]


@dataclass
class ConstraintSpec:
    """A full constraint specification over an ordered list of features."""

    features: List[FeatureSpec]
    interdependencies: List[Interdependency] = field(default_factory=list)
    # Numerical tolerance used by the validity checker for equality relations
    # and boundary comparisons.
    tol: float = 1e-6

    def __post_init__(self) -> None:
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names in ConstraintSpec")
        self.index: Dict[str, int] = {f.name: i for i, f in enumerate(self.features)}

    @property
    def n_features(self) -> int:
        return len(self.features)

    # -- derived arrays (vectorized enforcement) -------------------------------
    def mutable_mask(self) -> np.ndarray:
        return np.array([f.mutable for f in self.features], dtype=bool)

    def directions(self) -> np.ndarray:
        return np.array([f.direction for f in self.features], dtype=int)

    def lower_bounds(self) -> np.ndarray:
        return np.array([f.lo for f in self.features], dtype=float)

    def upper_bounds(self) -> np.ndarray:
        return np.array([f.hi for f in self.features], dtype=float)

    def integer_mask(self) -> np.ndarray:
        return np.array([f.integer for f in self.features], dtype=bool)


class ConstraintMask:
    """Enforces and checks a :class:`ConstraintSpec`.

    All methods work on batches ``X`` of shape (n_samples, n_features) and,
    where relevant, the original (clean) batch ``X0`` so that mutability and
    direction constraints are measured against the true starting point.
    """

    def __init__(self, spec: ConstraintSpec):
        self.spec = spec
        self._mutable = spec.mutable_mask()
        self._dir = spec.directions()
        self._lo = spec.lower_bounds()
        self._hi = spec.upper_bounds()
        self._int = spec.integer_mask()

    # ------------------------------------------------------------------ project
    def project(self, x_adv: np.ndarray, x_clean: np.ndarray) -> np.ndarray:
        """Project a perturbed batch back onto the feasible set.

        Enforcement order:
            1. immutable features are reset to their clean values;
            2. direction constraints clamp the *change* (increase/decrease only);
            3. box bounds clamp the resulting values;
            4. integer features are rounded;
            5. interdependency repairs are applied;
            6. box bounds re-applied (repairs may push out of range).

        The returned array is a new array (inputs are not mutated).
        """
        x_adv = np.asarray(x_adv, dtype=float)
        x_clean = np.asarray(x_clean, dtype=float)
        self._check_shape(x_adv)
        self._check_shape(x_clean)
        if x_adv.shape != x_clean.shape:
            raise ValueError("x_adv and x_clean must share shape")

        x = x_adv.copy()

        # 1. immutable features locked to clean values
        x[:, ~self._mutable] = x_clean[:, ~self._mutable]

        # 2. direction constraints on the delta
        delta = x - x_clean
        inc_only = self._dir == INCREASE_ONLY
        dec_only = self._dir == DECREASE_ONLY
        if inc_only.any():
            delta[:, inc_only] = np.clip(delta[:, inc_only], 0.0, None)
        if dec_only.any():
            delta[:, dec_only] = np.clip(delta[:, dec_only], None, 0.0)
        x = x_clean + delta

        # 3. box bounds
        x = self._clip_box(x)

        # 4. integer rounding
        if self._int.any():
            x[:, self._int] = np.rint(x[:, self._int])

        # 5. interdependency repairs
        for repair in self.spec.interdependencies:
            x = repair(x, self.spec.index)

        # 6. re-clip after repairs
        x = self._clip_box(x)
        return x

    def _clip_box(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self._lo, self._hi)

    # -------------------------------------------------------------- is_valid
    def is_valid(self, x_adv: np.ndarray, x_clean: np.ndarray) -> np.ndarray:
        """Return a boolean array (n_samples,): True == realizable sample.

        A sample is valid iff, relative to its clean counterpart, it:
            * leaves every immutable feature unchanged (within tol);
            * respects every direction constraint;
            * lies within every box bound;
            * satisfies every interdependency (checked by comparing the sample
              against its own repaired projection within tol).
        """
        x_adv = np.asarray(x_adv, dtype=float)
        x_clean = np.asarray(x_clean, dtype=float)
        self._check_shape(x_adv)
        self._check_shape(x_clean)
        tol = self.spec.tol
        n = x_adv.shape[0]
        valid = np.ones(n, dtype=bool)

        # immutable features unchanged
        if (~self._mutable).any():
            unchanged = np.all(
                np.abs(x_adv[:, ~self._mutable] - x_clean[:, ~self._mutable]) <= tol,
                axis=1,
            )
            valid &= unchanged

        # direction constraints
        delta = x_adv - x_clean
        inc_only = self._dir == INCREASE_ONLY
        dec_only = self._dir == DECREASE_ONLY
        if inc_only.any():
            valid &= np.all(delta[:, inc_only] >= -tol, axis=1)
        if dec_only.any():
            valid &= np.all(delta[:, dec_only] <= tol, axis=1)

        # box bounds
        within_box = np.all(
            (x_adv >= self._lo - tol) & (x_adv <= self._hi + tol), axis=1
        )
        valid &= within_box

        # interdependencies: a sample already satisfying them is unchanged by a
        # repair pass (up to tol). Only apply the interdependency repairs here
        # (not box/direction) so we isolate this class of violation.
        if self.spec.interdependencies:
            repaired = x_adv.copy()
            for repair in self.spec.interdependencies:
                repaired = repair(repaired, self.spec.index)
            consistent = np.all(np.abs(repaired - x_adv) <= 1e-4, axis=1)
            valid &= consistent

        return valid

    def valid_sample_rate(self, x_adv: np.ndarray, x_clean: np.ndarray) -> float:
        flags = self.is_valid(x_adv, x_clean)
        return float(np.mean(flags)) if flags.size else 0.0

    def _check_shape(self, x: np.ndarray) -> None:
        if x.ndim != 2 or x.shape[1] != self.spec.n_features:
            raise ValueError(
                f"Expected batch of shape (n, {self.spec.n_features}), got {x.shape}"
            )


# --------------------------------------------------------------------------- #
# Interdependency helpers (generic, dataset-agnostic building blocks).
# --------------------------------------------------------------------------- #
def sum_relation(total: str, parts: Sequence[str]) -> Interdependency:
    """Return a repair enforcing ``total == sum(parts)``.

    The total is recomputed from the parts (parts are treated as independently
    controllable, the derived total is made consistent). Used e.g. for
    ``total_packets = fwd_packets + bwd_packets``.
    """

    def repair(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        if total not in idx or any(p not in idx for p in parts):
            return x  # relation not applicable to this feature layout
        x = x.copy()
        x[:, idx[total]] = np.sum([x[:, idx[p]] for p in parts], axis=0)
        return x

    return repair


def ratio_relation(
    result: str, numerator: str, denominator: str, eps: float = 1e-9
) -> Interdependency:
    """Return a repair enforcing ``result == numerator / denominator``.

    Used for rate features (bytes/sec = total_bytes / flow_duration). A tiny eps
    guards against division by zero when duration is 0.
    """

    def repair(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        if any(c not in idx for c in (result, numerator, denominator)):
            return x
        x = x.copy()
        denom = x[:, idx[denominator]]
        x[:, idx[result]] = x[:, idx[numerator]] / np.where(
            np.abs(denom) < eps, eps, denom
        )
        return x

    return repair


def upper_bound_relation(
    feature: str, bound_features: Sequence[str], factor: float = 1.0
) -> Interdependency:
    """Clamp ``feature <= factor * prod(bound_features)``.

    Used e.g. for ``total_bytes <= total_packets * MSS`` (set factor = MSS and
    bound_features = ["total_packets"]).
    """

    def repair(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
        if feature not in idx or any(b not in idx for b in bound_features):
            return x
        x = x.copy()
        cap = np.full(x.shape[0], float(factor))
        for b in bound_features:
            cap = cap * x[:, idx[b]]
        x[:, idx[feature]] = np.minimum(x[:, idx[feature]], cap)
        return x

    return repair


# --------------------------------------------------------------------------- #
# Default / example specification.
# --------------------------------------------------------------------------- #
def build_default_spec(
    feature_names: Sequence[str],
    dataset: str = "cicids2017",
    immutable_keywords: Optional[Sequence[str]] = None,
    increase_only_keywords: Optional[Sequence[str]] = None,
) -> ConstraintSpec:
    """Build a reasonable, documented default ConstraintSpec for NIDS features.

    This uses *keyword heuristics* over column names to assign mutability and
    direction, which is a sensible starting point but MUST be reviewed against
    the exact dataset schema before running real experiments.

    Heuristics (domain assumptions from the paper, Section 2.5 / 4.4):
        * Features whose name contains a protocol/port/flag-type identifier are
          IMMUTABLE (an attacker cannot relabel the protocol or change which TCP
          flags *type* a feature counts without changing the attack itself).
        * Count/packet/byte/duration features are mutable, non-negative, and
          (by default) INCREASE_ONLY, reflecting that padding/delay can add but
          not remove already-sent traffic.
        * Everything else is mutable and FREE within [0, inf) unless it looks
          like it can be negative (e.g., contains "min"/"mean" of a signed
          quantity), in which case bounds are widened to (-inf, inf).

    TODO (dataset-specific, per Section 4.4):
        * Replace the keyword heuristics with an explicit per-column table for
          the chosen dataset (exact immutable columns, exact [lo, hi] ranges).
        * Add the concrete interdependency relations that hold for the dataset's
          real column names (the examples below are wired only if the matching
          columns are present).
    """
    immutable_keywords = list(
        immutable_keywords
        if immutable_keywords is not None
        else [
            "protocol",
            "proto",
            "port",  # src/dst port -- attacker-fixed for a given service
            "flag",  # TCP flag-type indicator columns
            "service",
            "state",
        ]
    )
    increase_only_keywords = list(
        increase_only_keywords
        if increase_only_keywords is not None
        else ["packet", "pkt", "byte", "duration", "len", "count", "tot", "delay"]
    )

    specs: List[FeatureSpec] = []
    for name in feature_names:
        lname = name.lower()
        is_immutable = any(k in lname for k in immutable_keywords)
        if is_immutable:
            # Immutable: locked to clean value; wide bounds (never clipped since
            # delta is forced to 0 anyway).
            specs.append(
                FeatureSpec(name=name, mutable=False, direction=FREE, lo=-np.inf, hi=np.inf)
            )
            continue

        direction = INCREASE_ONLY if any(k in lname for k in increase_only_keywords) else FREE
        # Signed statistics (differences, means of signed quantities) may be
        # negative; keep them unbounded below. Counts/bytes/durations stay >= 0.
        can_be_negative = any(k in lname for k in ["diff", "delta"]) and not any(
            k in lname for k in ["packet", "pkt", "byte", "count", "tot"]
        )
        lo = -np.inf if can_be_negative else 0.0
        integer = any(k in lname for k in ["packet", "pkt", "count", "flag", "tot_pkt"]) and (
            "rate" not in lname and "sec" not in lname and "mean" not in lname
        )
        specs.append(
            FeatureSpec(name=name, mutable=True, direction=direction, lo=lo, hi=np.inf, integer=integer)
        )

    interdeps: List[Interdependency] = []
    # Wire example relations only if plausible columns exist. These names are
    # illustrative; TODO: replace with the dataset's exact column names.
    lut = {n.lower(): n for n in feature_names}

    def find(*cands: str) -> Optional[str]:
        for c in cands:
            if c in lut:
                return lut[c]
        return None

    total_pkts = find("total packets", "tot packets", "total_packets")
    fwd_pkts = find("total fwd packets", "fwd packets", "total_fwd_packets")
    bwd_pkts = find("total backward packets", "bwd packets", "total_bwd_packets")
    if total_pkts and fwd_pkts and bwd_pkts:
        interdeps.append(sum_relation(total_pkts, [fwd_pkts, bwd_pkts]))

    return ConstraintSpec(features=specs, interdependencies=interdeps)


def build_synthetic_spec() -> ConstraintSpec:
    """A tiny, fully-specified spec used by unit tests and demos.

    Feature layout (6 features):
        0 protocol        - IMMUTABLE (protocol identifier)
        1 dst_port        - IMMUTABLE (service port)
        2 fwd_packets     - mutable, INCREASE_ONLY, integer, >= 0
        3 bwd_packets     - mutable, INCREASE_ONLY, integer, >= 0
        4 total_packets   - derived: == fwd_packets + bwd_packets
        5 mean_iat        - mutable, FREE, may be negative-ish but >= 0 here
    """
    features = [
        FeatureSpec("protocol", mutable=False, direction=FREE, lo=-np.inf, hi=np.inf),
        FeatureSpec("dst_port", mutable=False, direction=FREE, lo=-np.inf, hi=np.inf),
        FeatureSpec("fwd_packets", mutable=True, direction=INCREASE_ONLY, lo=0.0, hi=np.inf, integer=True),
        FeatureSpec("bwd_packets", mutable=True, direction=INCREASE_ONLY, lo=0.0, hi=np.inf, integer=True),
        FeatureSpec("total_packets", mutable=True, direction=INCREASE_ONLY, lo=0.0, hi=np.inf, integer=True),
        FeatureSpec("mean_iat", mutable=True, direction=FREE, lo=0.0, hi=np.inf, integer=False),
    ]
    interdeps = [sum_relation("total_packets", ["fwd_packets", "bwd_packets"])]
    return ConstraintSpec(features=features, interdependencies=interdeps)
