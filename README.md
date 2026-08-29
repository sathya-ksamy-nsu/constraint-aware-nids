# Constraint-Aware Adversarial NIDS

MIT-licensed **evaluation harness** for constraint-aware adversarial attacks on ML-based network intrusion detection.

It compares gradient (FGSM/PGD), decision-based (HopSkipJump), and GAN attacks in **unconstrained** vs **constrained** regimes, and reports attack success rate, perturbation size, and **valid-sample rate**.

Manuscripts, proposals, and arXiv sources live in a separate private repository. This repo is **code, config, and tests only**.

**Author:** Sathyaraj Kolandasamy  
**Affiliation:** Nova Southeastern University

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -v
python experiments/run_experiment.py --synthetic --model mlp --attack all
```

Raw CICIDS-2017 / UNSW-NB15 files are **not** included. See [`data/README.md`](data/README.md), then:

```powershell
python experiments/run_experiment.py --model mlp --attack all
```

## Layout

| Path | Contents |
| --- | --- |
| `src/` | Constraint mask, models, attacks, metrics |
| `experiments/` | CLI for the experiment matrix |
| `tests/` | Unit tests (no dataset required) |
| `config.yaml` | Seeds, models, attack settings |
| `data/README.md` | How to obtain public datasets |

## License

MIT. Do not relicense third-party datasets.
