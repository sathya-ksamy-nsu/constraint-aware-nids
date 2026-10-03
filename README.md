# MetroCon 2026 — anonymous reproduction artifacts (Topic 1)

Anonymous package of **data, scripts, and materials** for the constraint-aware adversarial NIDS study (double-blind review).

**Paper sources and PDFs are intentionally excluded** from this folder.

## Contents

| Path | What it is |
| --- | --- |
| `src/` | Constraint mask, models, ART attack wrappers, metrics, defense |
| `experiments/run_experiment.py` | One-command experiment CLI |
| `scripts/` | Dataset download helper, freeze indices, run metadata |
| `tests/` | Unit tests (no raw dataset required) |
| `config.yaml` | Seeds, models, attack/defense settings |
| `requirements.txt` | Python dependencies |
| `data/README.md` | How to obtain CICIDS-2017 / UNSW-NB15 (raw CSVs not included) |
| `results/splits/` | Fixed train/val/test index files (`*_seed42.npz`) |
| `results/real/full/` | Full-corpus freeze JSON used for Section 6 tables |
| `results/real/*.json` | Earlier 80k-extract freeze (historical; not Section 6) |
| `results/DATA_FREEZE.md` | Freeze protocol notes |
| `results/run_meta.json` | Environment / run metadata snapshot (paths redacted) |
| `RESULTS.md` | Human-readable result summary |
| `LICENSE` | MIT (anonymous copyright for review) |
| `CITATION.cff` | Anonymous citation stub for review |

## Not included

- Raw CICIDS-2017 / UNSW-NB15 CSVs or PCAPs (`data/raw/` stays local)
- Paper Markdown / LaTeX / PDF / BibTeX
- Synthetic smoke JSON dumps
- Author names, emails, affiliations, or personal repository URLs

## Quick start

```bash
cd metrocon-artifacts
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests/ -v
# After placing raw CSVs per data/README.md:
python experiments/run_experiment.py --model mlp --attack all --output-dir results/real/full
```

## Artifact boundary

Released materials are **code, config, constraint specification, fixed split indices, tests, and freeze result JSON**. Raw traffic is not redistributed.
