"""CLI: write results/run_meta.json (Python, packages, CPU/GPU).

Run from the topic-1 project root:

    python scripts/write_run_meta.py
    python scripts/write_run_meta.py --out results/run_meta.json --label environment
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Record harness environment metadata.")
    p.add_argument(
        "--out",
        default=os.path.join("results", "run_meta.json"),
        help="Output JSON path (default: results/run_meta.json).",
    )
    p.add_argument(
        "--label",
        default="environment",
        help="Record label (e.g. environment, SYNTHETIC / PIPELINE VALIDATION).",
    )
    args = p.parse_args(argv)

    from src.run_meta import write_run_meta

    path = write_run_meta(args.out, label=args.label)
    print(f"[info] wrote run metadata: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
