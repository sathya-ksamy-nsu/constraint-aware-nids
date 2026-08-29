# Experiment Run Report

**Project:** Constraint-Aware Adversarial-NIDS Evaluation Harness
**Run date:** 2026-07-31
**Mode:** `--synthetic` (pipeline validation)

> **Read this first.** These numbers were produced on **synthetic Gaussian-blob
> data**, not on a real NIDS dataset (CICIDS-2017 / UNSW-NB15). They exist to
> **validate that the harness runs end-to-end and that the constraint / validity
> accounting behaves as designed.** They are **not** scientific findings and the
> absolute attack-success numbers should not be cited as evidence about real
> intrusion detectors. To generate reportable results, drop a real dataset into
> `data/raw/` and re-run without `--synthetic` (see "Reproduction" below).

---

## 1. Environment

| Component | Version |
|---|---|
| OS | Windows 10 |
| Python | 3.13.3 |
| PyTorch | 2.10.0+cpu |
| Adversarial Robustness Toolbox (ART) | 1.20.1 |
| scikit-learn | 1.8.0 |
| Compute | CPU only |

## 2. Unit tests

```
python -m pytest tests/ -v
```

**Result: 27 passed in 0.31s.**

- `tests/test_constraints.py` — 15 passed (immutability reset, increase-only
  projection, range / non-negativity clipping, integer rounding, interdependency
  repair, valid-sample-rate accounting, default-spec wiring).
- `tests/test_metrics.py` — 12 passed (clean accuracy, attack-success-rate
  denominators and edge cases, perturbation-size math with/without mask,
  valid-sample-rate, summary keys).

## 3. Target model: MLP (full matrix)

```
python experiments/run_experiment.py --synthetic --model mlp --attack all
```

This sweeps **4 attacks x 2 constraint regimes x 2 defense states = 16 cells**.
Artifacts written: `results/mlp_all_20260731-195454.json` and `.csv`.

Metric definitions:

- **ASR** — attack success rate: fraction of originally-detected malicious flows
  that evade the detector after perturbation. Higher = stronger attack.
- **Valid-rate** — fraction of the adversarial examples that actually satisfy the
  domain constraint mask (protocol immutable, byte/packet counts non-negative and
  increase-only, feature interdependencies consistent). Low valid-rate = the
  "evasions" are **not realizable network traffic**.
- **Mean L2** — average L2 perturbation size over attacked samples.

### 3a. Undefended MLP (`defense = none`)

| Attack | Regime | ASR | Valid-rate | Mean L2 |
|---|---|---:|---:|---:|
| FGSM | unconstrained | 0.000 | 0.327 | 0.346 |
| FGSM | constrained | 0.000 | 1.000 | 0.480 |
| PGD | unconstrained | 0.000 | 0.327 | 0.346 |
| PGD | constrained | 0.000 | 1.000 | 0.480 |
| HopSkipJump | unconstrained | **1.000** | **0.000** | 2.923 |
| HopSkipJump | constrained | **0.027** | **1.000** | 2.325 |
| GAN | unconstrained | 0.000 | 0.380 | 0.133 |
| GAN | constrained | 0.000 | 1.000 | 0.325 |

### 3b. Adversarially-trained MLP (`defense = adv_training`)

| Attack | Regime | ASR | Valid-rate | Mean L2 |
|---|---|---:|---:|---:|
| FGSM | unconstrained | 0.007 | 0.327 | 0.346 |
| FGSM | constrained | 0.000 | 1.000 | 0.480 |
| PGD | unconstrained | 0.007 | 0.327 | 0.346 |
| PGD | constrained | 0.000 | 1.000 | 0.479 |
| HopSkipJump | unconstrained | **1.000** | **0.000** | 2.402 |
| HopSkipJump | constrained | **0.101** | **1.000** | 1.994 |
| GAN | unconstrained | 0.000 | 0.380 | 0.133 |
| GAN | constrained | 0.000 | 1.000 | 0.324 |

## 4. What the run demonstrates (mechanics, not findings)

1. **The central thesis mechanism works.** The decision-based black-box attack
   (HopSkipJump) reaches **ASR = 1.000 with valid-rate = 0.000** in the
   *unconstrained* regime — i.e. it "defeats" the detector entirely, but **none**
   of those adversarial flows are valid network traffic. Once the constraint mask
   is enforced (*constrained* regime, valid-rate = 1.000 by construction), ASR
   collapses to **0.027** (undefended) / **0.101** (defended). This is exactly the
   paper's argument: **unconstrained ASR dramatically overstates the real-world
   threat**, and the gap between the two regimes is the quantity of interest.
2. **Validity accounting is wired correctly.** Unconstrained gradient/GAN attacks
   land in the 0.33-0.38 valid-rate band (perturbations routinely violate the
   mask), while every constrained-regime cell reports valid-rate = 1.000, meaning
   the projection step is enforced before scoring.
3. **The defense hook executes** (adversarial training changes the numbers) and
   the full cross-product matrix is produced and serialized to JSON + CSV.

> On this easy synthetic set FGSM/PGD show ASR = 0 (the MLP is confidently correct
> at the configured epsilon), so gradient-attack *potency* is not exercised here —
> that requires real data. Likewise the small constrained-HopSkipJump ASR rise
> under adversarial training (0.027 -> 0.101) is a synthetic-data artifact, not a
> result.

## 5. Target model: Random Forest (status)

- **FGSM / PGD are not applicable** to the Random Forest target: it is
  non-differentiable, so gradient attacks have no gradients to use. (The harness
  correctly restricts RF to black-box attacks.)
- **HopSkipJump / GAN on RF were attempted but not completed** in this session:
  decision-based black-box search issues hundreds of thousands of single-sample
  queries against the forest, and the run was cut short by a filesystem/shell
  lock on this machine before results were written. A follow-up fix was applied
  (`random_forest.n_jobs: 1`, since per-query joblib parallelism is pure overhead
  for these tiny single-sample predicts) and the RF black-box cells should be
  re-run to complete the matrix.

## 6. Harness fixes applied during this run

1. **`src/models.py` — float32 input cast in the MLP.** ART's HopSkipJump queries
   the estimator with float64 arrays, which hit a `Double vs Float` dtype mismatch
   in the PyTorch `Linear` layers. Added a leading cast module so the network
   accepts any input dtype. (Without this, `--model mlp --attack all` crashes at
   the HopSkipJump cell.)
2. **`config.yaml` — `random_forest.n_jobs: 1`.** Black-box attacks make many tiny
   single-sample predictions where joblib parallelism only adds overhead (and
   floods warnings); single-threaded is faster for this access pattern.

## 7. Reproduction

```bash
pip install -r requirements.txt

# Unit tests
python -m pytest tests/ -v

# Pipeline validation on synthetic data (what this report used)
python experiments/run_experiment.py --synthetic --model mlp --attack all
python experiments/run_experiment.py --synthetic --model random_forest --attack hopskipjump
python experiments/run_experiment.py --synthetic --model random_forest --attack gan

# Real results: place CICIDS-2017 / UNSW-NB15 under data/raw/ (see data/README.md),
# then drop the --synthetic flag.
python experiments/run_experiment.py --model mlp --attack all
```

Raw `results/*.json` and `results/*.csv` are intentionally git-ignored; regenerate
them locally with the commands above.
