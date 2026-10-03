"""Collect and persist environment metadata for a harness run.

Writes ``results/run_meta.json`` with Python, package, and CPU/GPU details.
This module fabricates no experimental metrics; it only records the machine
that produced a run. Heavy packages are imported opportunistically so unit
tests can still collect a partial record without torch or ART.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

DIRECT_PACKAGES = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scikit-learn", "sklearn"),
    ("torch", "torch"),
    ("adversarial-robustness-toolbox", "art"),
    ("matplotlib", "matplotlib"),
    ("pyyaml", "yaml"),
    ("pytest", "pytest"),
)


def _package_version(import_name: str) -> Optional[str]:
    try:
        mod = __import__(import_name)
    except Exception:
        return None
    return getattr(mod, "__version__", "unknown")


def collect_run_meta(
    *,
    label: str = "environment",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable environment record."""
    packages: Dict[str, Any] = {}
    for dist_name, import_name in DIRECT_PACKAGES:
        version = _package_version(import_name)
        packages[dist_name] = version if version is not None else "unavailable"

    cpu_gpu: Dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "platform": platform.platform(),
        "torch_device": "unavailable",
        "cuda_available": False,
        "gpu_name": None,
        "torch_cuda_version": None,
    }
    try:
        import torch  # type: ignore

        cpu_gpu["torch_device"] = "cuda" if torch.cuda.is_available() else "cpu"
        cpu_gpu["cuda_available"] = bool(torch.cuda.is_available())
        cpu_gpu["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            cpu_gpu["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    meta: Dict[str, Any] = {
        "label": label,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "packages": packages,
        "cpu_gpu": cpu_gpu,
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_run_meta(
    path: str,
    *,
    label: str = "environment",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Write ``collect_run_meta`` to ``path`` and return that path."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    meta = collect_run_meta(label=label, extra=extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return path
