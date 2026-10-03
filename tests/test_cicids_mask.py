"""CICIDS-2017 explicit constraint table."""
from src.cicids_mask import cicids2017_feature_spec, cicids2017_interdependencies
from src.constraints import build_default_spec


def test_destination_port_immutable():
    spec = cicids2017_feature_spec("Destination Port")
    assert spec is not None
    assert spec.mutable is False


def test_fwd_packets_increase_only():
    spec = cicids2017_feature_spec("Total Fwd Packets")
    assert spec is not None
    assert spec.mutable is True
    assert spec.direction == 1
    assert spec.integer is True


def test_bwd_packets_immutable_victim_side():
    spec = cicids2017_feature_spec("Total Backward Packets")
    assert spec is not None
    assert spec.mutable is False


def test_unknown_column_returns_none():
    assert cicids2017_feature_spec("not_a_cicids_column") is None


def test_build_default_spec_uses_explicit_table():
    names = ["Destination Port", "Total Fwd Packets", "Flow Duration", "Flow Bytes/s"]
    spec = build_default_spec(names, dataset="cicids2017")
    assert spec.features[0].mutable is False
    assert spec.features[1].direction == 1
    assert spec.fingerprint()


def test_interdeps_wire_subflow_and_rates():
    names = [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Subflow Fwd Packets",
        "Flow Duration",
        "Flow Bytes/s",
        "Flow Packets/s",
    ]
    deps = cicids2017_interdependencies(names)
    assert len(deps) >= 3
