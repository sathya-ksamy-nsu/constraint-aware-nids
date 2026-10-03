"""Explicit CICIDS-2017 CICFlowMeter constraint table (Section 4.4).

Column names match the official MachineLearningCSV export after whitespace
stripping. Unlisted columns fall back to the keyword heuristics in
``constraints.build_default_spec``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .constraints import (
    FREE,
    INCREASE_ONLY,
    FeatureSpec,
    Interdependency,
    ratio_relation,
    sum_relation,
    upper_bound_relation,
)

# Victim-/destination-side counters an off-path packet sender cannot set.
_BWD_IMMUTABLE: Set[str] = {
    "Total Backward Packets",
    "Total Length of Bwd Packets",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Bwd Header Length",
    "Bwd Packets/s",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    "Avg Bwd Segment Size",
    "Init_Win_bytes_backward",
}

# Protocol/port/flag-type / handshake fields.
_ID_IMMUTABLE: Set[str] = {
    "Destination Port",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Init_Win_bytes_forward",
    "min_seg_size_forward",
}

CICIDS2017_IMMUTABLE: Set[str] = _BWD_IMMUTABLE | _ID_IMMUTABLE

# Derived rates/ratios: mutable so the attack can propose values, then repaired.
CICIDS2017_FREE_NONNEG: Set[str] = {
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packets/s",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
}

CICIDS2017_INCREASE_ONLY: Set[str] = {
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Fwd Header Length",
    "Fwd Header Length.1",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "act_data_pkt_fwd",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
}

_INTEGER_NAMES: Set[str] = {
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Header Length.1",
    "Min Packet Length",
    "Max Packet Length",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
}


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _lookup() -> Dict[str, str]:
    names = CICIDS2017_IMMUTABLE | CICIDS2017_FREE_NONNEG | CICIDS2017_INCREASE_ONLY
    return {_norm(n): n for n in names}


def cicids2017_feature_spec(name: str) -> Optional[FeatureSpec]:
    """Return an explicit FeatureSpec if ``name`` is a known CICIDS-2017 column."""
    canonical = _lookup().get(_norm(name))
    if canonical is None:
        return None
    if canonical in CICIDS2017_IMMUTABLE:
        return FeatureSpec(
            name=name,
            mutable=False,
            direction=FREE,
            lo=-np.inf,
            hi=np.inf,
            integer=canonical in _INTEGER_NAMES,
        )
    if canonical in CICIDS2017_FREE_NONNEG:
        return FeatureSpec(
            name=name,
            mutable=True,
            direction=FREE,
            lo=0.0,
            hi=np.inf,
            integer=False,
        )
    # increase-only attacker-controlled quantities
    return FeatureSpec(
        name=name,
        mutable=True,
        direction=INCREASE_ONLY,
        lo=0.0,
        hi=np.inf,
        integer=canonical in _INTEGER_NAMES,
    )


def cicids2017_interdependencies(feature_names: Sequence[str]) -> List[Interdependency]:
    """Wire CICFlowMeter equalities/rates using the dataset's real column names."""
    lut = {_norm(n): n for n in feature_names}

    def find(*cands: str) -> Optional[str]:
        for c in cands:
            hit = lut.get(_norm(c))
            if hit is not None:
                return hit
        return None

    interdeps: List[Interdependency] = []

    fwd_pkts = find("Total Fwd Packets")
    bwd_pkts = find("Total Backward Packets")
    fwd_len = find("Total Length of Fwd Packets")
    bwd_len = find("Total Length of Bwd Packets")
    duration = find("Flow Duration")
    sub_fwd_pkts = find("Subflow Fwd Packets")
    sub_fwd_bytes = find("Subflow Fwd Bytes")
    sub_bwd_pkts = find("Subflow Bwd Packets")
    sub_bwd_bytes = find("Subflow Bwd Bytes")
    avg_fwd = find("Avg Fwd Segment Size")
    avg_bwd = find("Avg Bwd Segment Size")
    avg_size = find("Average Packet Size")
    flow_bps = find("Flow Bytes/s")
    flow_pps = find("Flow Packets/s")
    fwd_pps = find("Fwd Packets/s")
    bwd_pps = find("Bwd Packets/s")
    down_up = find("Down/Up Ratio")
    hdr = find("Fwd Header Length")
    hdr_dup = find("Fwd Header Length.1")

    if sub_fwd_pkts and fwd_pkts:
        interdeps.append(sum_relation(sub_fwd_pkts, [fwd_pkts]))
    if sub_fwd_bytes and fwd_len:
        interdeps.append(sum_relation(sub_fwd_bytes, [fwd_len]))
    if avg_fwd and fwd_len and fwd_pkts:
        interdeps.append(ratio_relation(avg_fwd, fwd_len, fwd_pkts))
    if hdr and hdr_dup:
        interdeps.append(sum_relation(hdr_dup, [hdr]))

    # Clamp attacker-controlled payloads before recomputing rates.
    if fwd_len and fwd_pkts:
        interdeps.append(upper_bound_relation(fwd_len, [fwd_pkts], factor=1500.0))

    # CICFlowMeter stores Flow Duration in microseconds.
    usec = 1e-6
    if flow_bps and fwd_len and bwd_len and duration:

        def repair_bps(x: np.ndarray, idx: Dict[str, int], *, _usec=usec) -> np.ndarray:
            if any(c not in idx for c in (flow_bps, fwd_len, bwd_len, duration)):
                return x
            x = x.copy()
            total_bytes = x[:, idx[fwd_len]] + x[:, idx[bwd_len]]
            seconds = np.maximum(x[:, idx[duration]] * _usec, 1e-9)
            x[:, idx[flow_bps]] = total_bytes / seconds
            return x

        interdeps.append(repair_bps)

    if flow_pps and fwd_pkts and bwd_pkts and duration:

        def repair_pps(x: np.ndarray, idx: Dict[str, int], *, _usec=usec) -> np.ndarray:
            if any(c not in idx for c in (flow_pps, fwd_pkts, bwd_pkts, duration)):
                return x
            x = x.copy()
            total_pkts = x[:, idx[fwd_pkts]] + x[:, idx[bwd_pkts]]
            seconds = np.maximum(x[:, idx[duration]] * _usec, 1e-9)
            x[:, idx[flow_pps]] = total_pkts / seconds
            return x

        interdeps.append(repair_pps)

    if fwd_pps and fwd_pkts and duration:

        def repair_fwd_pps(x: np.ndarray, idx: Dict[str, int], *, _usec=usec) -> np.ndarray:
            if any(c not in idx for c in (fwd_pps, fwd_pkts, duration)):
                return x
            x = x.copy()
            seconds = np.maximum(x[:, idx[duration]] * _usec, 1e-9)
            x[:, idx[fwd_pps]] = x[:, idx[fwd_pkts]] / seconds
            return x

        interdeps.append(repair_fwd_pps)

    if down_up and fwd_pkts and bwd_pkts:
        interdeps.append(ratio_relation(down_up, bwd_pkts, fwd_pkts))

    if avg_size and fwd_len and bwd_len and fwd_pkts and bwd_pkts:

        def repair_avg_size(x: np.ndarray, idx: Dict[str, int]) -> np.ndarray:
            need = (avg_size, fwd_len, bwd_len, fwd_pkts, bwd_pkts)
            if any(c not in idx for c in need):
                return x
            x = x.copy()
            tot_len = x[:, idx[fwd_len]] + x[:, idx[bwd_len]]
            tot_pkts = np.maximum(x[:, idx[fwd_pkts]] + x[:, idx[bwd_pkts]], 1.0)
            x[:, idx[avg_size]] = tot_len / tot_pkts
            return x

        interdeps.append(repair_avg_size)

    return interdeps
