# Constraint-Aware Adversarial NIDS

Open evaluation **protocol and harness** for the preprint:

**How Realistic Are “Successful” Evasion Attacks? A Constraint-Aware Reproducible Comparison of Adversarial Attacks Against ML-Based Network Intrusion Detection**

**Author:** Sathyaraj Kolandasamy  
**Affiliation:** Nova Southeastern University, College of Computing, AI, and Cybersecurity  
**License:** MIT

Hypothesis: constraining adversarial perturbations to semantically valid, functionality-preserving flow features **lowers reported attack success rate (ASR)** and can **reorder** attack/defense rankings versus unconstrained feature-space evaluation.

> Synthetic pipeline-validation numbers in `RESULTS.md` are **not** findings about real NIDS. CICIDS-2017 / UNSW-NB15 results are forthcoming. Raw datasets are **not** included.

## Repository contents

| Path | What it is |
| --- | --- |
| [`arxiv-preprint.md`](arxiv-preprint.md) | Preprint manuscript (Markdown) |
| [`latex/main.tex`](latex/main.tex) | **arXiv upload:** LaTeX source (compile to PDF) |
| [`ARXIV.md`](ARXIV.md) | How to submit to arXiv |
| [`src/`](src/) | Constraint mask, models, ART attacks, metrics |
| [`experiments/run_experiment.py`](experiments/run_experiment.py) | One-command experiment matrix |
| [`tests/`](tests/) | NumPy-only unit tests (no datasets required) |
| [`data/README.md`](data/README.md) | How to obtain CICIDS-2017 / UNSW-NB15 |
| [`RESULTS.md`](RESULTS.md) | Synthetic harness validation log |
| [`CITATION.cff`](CITATION.cff) | Cite this repository |

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -v
python experiments/run_experiment.py --synthetic --model mlp --attack all
```

After placing CICIDS/UNSW CSVs under `data/raw/` (see `data/README.md`):

```powershell
python experiments/run_experiment.py --model mlp --attack all
```

## Cite

See [`CITATION.cff`](CITATION.cff). After the arXiv id is assigned, add it there and in `latex/main.tex`.
