"""Environment metadata helper (no torch/ART required)."""
from src.run_meta import collect_run_meta, write_run_meta


def test_collect_run_meta_has_required_keys():
    meta = collect_run_meta(label="environment")
    assert meta["label"] == "environment"
    assert "python_version" in meta
    assert "packages" in meta
    assert "cpu_gpu" in meta
    assert "numpy" in meta["packages"]
    assert meta["packages"]["numpy"] not in (None, "unavailable")


def test_write_run_meta_roundtrip(tmp_path):
    path = tmp_path / "run_meta.json"
    written = write_run_meta(str(path), label="environment")
    assert written == str(path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert '"python_version"' in text
    assert '"scikit-learn"' in text
