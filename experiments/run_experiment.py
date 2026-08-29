"""CLI entry point for the constraint-aware adversarial-NIDS harness.

Reads ``config.yaml``, wires data -> model -> attacks -> defense -> metrics, and
writes results to ``results/``. Nothing here fabricates numbers; if the real
dataset is missing you can exercise the full pipeline on synthetic data with
``--synthetic`` (clearly labeled, not experimental results).

Examples (PowerShell, from the topic-1 project root):

    # Full matrix on the configured dataset with the MLP target:
    python experiments/run_experiment.py --model mlp

    # A single attack, constrained only, no defense, on synthetic data (smoke):
    python experiments/run_experiment.py --synthetic --attack pgd --constrained --no-defense

    # Random-forest target, black-box attack:
    python experiments/run_experiment.py --model random_forest --attack hopskipjump

Run ``python experiments/run_experiment.py --help`` for all flags.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make ``src`` importable when run as a script from the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Constraint-aware adversarial-NIDS evaluation harness."
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--dataset",
        default=None,
        help="Override dataset name (cicids2017 | unsw-nb15).",
    )
    p.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "random_forest"],
        help="Target model to attack.",
    )
    p.add_argument(
        "--attack",
        default="all",
        help="Attack name (fgsm|pgd|hopskipjump|zoo|gan) or 'all'.",
    )
    reg = p.add_mutually_exclusive_group()
    reg.add_argument(
        "--constrained",
        dest="constrained",
        action="store_true",
        help="Run constrained regime only (default: run both regimes).",
    )
    reg.add_argument(
        "--unconstrained",
        dest="unconstrained",
        action="store_true",
        help="Run unconstrained regime only.",
    )
    p.add_argument(
        "--no-defense",
        dest="no_defense",
        action="store_true",
        help="Skip the adversarial-training defense sweep.",
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use fabricated synthetic data (pipeline smoke test; NOT results).",
    )
    p.add_argument("--seed", type=int, default=None, help="Override the RNG seed.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.dataset:
        cfg.setdefault("dataset", {})["name"] = args.dataset
    if args.seed is not None:
        cfg["seed"] = args.seed

    from src import evaluate as E
    from src.models import set_global_seeds

    set_global_seeds(int(cfg.get("seed", 42)))

    # --- load data ---
    if args.synthetic:
        from src.data_loader import make_synthetic_dataset

        print("[info] Using SYNTHETIC data -- results are for pipeline testing only.")
        data = make_synthetic_dataset(seed=int(cfg.get("seed", 42)))
    else:
        from src.data_loader import load_dataset, DataNotFoundError

        try:
            data = load_dataset(cfg)
        except DataNotFoundError as e:
            print(f"[error] {e}", file=sys.stderr)
            print(
                "[hint] Pass --synthetic to smoke-test the pipeline without real "
                "data, or follow data/README.md to obtain the dataset.",
                file=sys.stderr,
            )
            return 2

    # --- resolve attack list ---
    if args.attack == "all":
        attack_names = cfg.get("attacks", {}).get("enabled", ["pgd"])
    else:
        attack_names = [args.attack]

    include_defense = not args.no_defense

    # If a single regime was requested, restrict the matrix accordingly by
    # toggling the config default and filtering afterward.
    records = E.run_matrix(
        model_name=args.model,
        data=data,
        cfg=cfg,
        attack_names=attack_names,
        include_defense=include_defense,
    )

    if args.constrained:
        records = [r for r in records if r["constrained"]]
    elif args.unconstrained:
        records = [r for r in records if not r["constrained"]]

    # --- report + persist ---
    print(f"\n[info] Completed {len(records)} matrix cell(s):")
    for r in records:
        print(
            f"  attack={r['attack']:<11} constrained={str(r['constrained']):<5} "
            f"defense={r['defense']:<12} ASR={r.get('asr', float('nan')):.3f} "
            f"valid_rate={r.get('valid_sample_rate', float('nan')):.3f} "
            f"L2={r.get('l2_mean', float('nan')):.3f}"
        )

    written = E.write_results(records, cfg, tag=f"{args.model}_{args.attack}")
    for fmt, path in written.items():
        print(f"[info] wrote {fmt}: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
