# How Realistic Are "Successful" Evasion Attacks? A Constraint-Aware Reproducible Comparison of Adversarial Attacks Against ML-Based Network Intrusion Detection

**Author:** Sathyaraj Kolandasamy  
**Affiliation:** Nova Southeastern University, College of Computing, AI, and Cybersecurity  
**Correspondence:** via [github.com/sathya-ksamy-nsu/constraint-aware-nids](https://github.com/sathya-ksamy-nsu/constraint-aware-nids)

**arXiv intended categories:** `cs.CR` (primary), `cs.LG` (cross-list)  
**Version:** preprint v0.1 — 2026-08-29  
**Code:** this repository (MIT License)

> **Preprint status.** This is a **Phase 1 arXiv preprint** of an evaluation *protocol and open harness*. The constraint mask, threat model, related-work synthesis, and experiment design are complete. **Benchmark results on CICIDS-2017 / UNSW-NB15 are not yet reported.** Section 6 includes a *pipeline-validation* run on **synthetic data only**; those numbers demonstrate that the harness accounts for validity correctly — they are **not** findings about real intrusion detectors. Replace Section 6 with CICIDS/UNSW measurements before workshop/journal submission.

---

## Abstract

Machine-learning-based network intrusion detection systems (NIDS) are widely evaluated under adversarial evasion, and many papers report very high attack success rates (ASR). Surveys from 2023–2024 warn that those successes are often **unrealistic**: attacks perturb flow features independently, violating protocol semantics and feature interdependencies that a packet-sending adversary cannot break. This preprint contributes a **constraint-aware evaluation protocol** and an **open harness** to measure that gap. We formalize a reusable *realistic-perturbation mask* (mutability, direction, range, and interdependency constraints), compare gradient-based (FGSM/PGD), decision-based (HopSkipJump), and generative (GAN) attacks in **unconstrained** vs **constrained** regimes, and report ASR together with L2/L∞ size and a **valid-sample rate**. The hypothesis, to be tested on CICIDS-2017 or UNSW-NB15, is that enforcing semantically valid, functionality-preserving perturbations **lowers reported ASR and can reorder attack/defense rankings**. A synthetic end-to-end run of the harness shows the intended mechanism: unconstrained HopSkipJump reached ASR = 1.00 with valid-sample rate = 0.00, while the constrained regime forced valid-rate = 1.00 and collapsed ASR to 0.027 (undefended MLP). Those figures are **pipeline validation, not NIDS evidence**. The artifact is the protocol, mask, and one-command matrix — intended as a citable baseline for follow-on work and for a forthcoming workshop version (AI-SEC 2027).

---

## 1. Introduction

Modern networks overwhelm signature-based intrusion detection. ML-based NIDS learn statistical regularities of benign and malicious flows and are now common in research and production stacks (He et al., 2023). Their decision boundary is therefore a security surface, not only an accuracy metric.

Adversarial examples show that small perturbations can flip a classifier (Goodfellow et al., 2015; Madry et al., 2018). Against a NIDS, evasion means malicious traffic is labeled benign. The literature reports many such attacks with high ASR (He et al., 2023; Sharma & Chen, 2024).

A structural flaw remains: most attacks operate in **feature space**, treating each CICFlowMeter-style coordinate as independently mutable under a norm budget. A flow vector is a *derived summary* of packets. An attacker who can only send packets cannot independently set inter-arrival statistics, durations, and byte totals to inconsistent values (Alhussien et al., 2024; Ennaji et al., 2024). Unconstrained ASR can therefore overstate the threat, and defenses can be tuned against impossible traffic.

**Research question.** When perturbations against an ML-based NIDS are restricted to semantically valid, functionality-preserving changes, how do ASR and the ranking of attacks and defenses change relative to unconstrained feature-space evaluation?

**Hypothesis.** Constraining perturbations **lowers reported ASR** and **reorders** attack/defense rankings versus unconstrained perturbations.

**Contributions.**

1. A formal, reusable **constraint specification** (realistic-perturbation mask).
2. A **reproducible harness** (identical preprocessing/splits, ART-based attacks, valid-sample accounting).
3. A controlled unconstrained-vs-constrained **experiment design** spanning white-box, black-box, and generative attacks, with and without adversarial training.
4. A related-work table annotated by whether prior attacks respect domain constraints.
5. An **open implementation** (this repository) with unit tests and a documented synthetic pipeline-validation run.

Section 2 states the threat model. Section 3 reviews related work. Section 4 specifies the protocol. Section 5 records the experimental setup. Section 6 reports pipeline validation (synthetic) and the pending real-data matrix. Sections 7–10 discuss implications, limits, conclusions, and reproducibility.

---

## 2. Background and threat model

### 2.1 ML-based NIDS pipeline

Packets are grouped into flows (five-tuple). Feature extraction (e.g. CICFlowMeter) yields counts, durations, flag tallies, and rates. After preprocessing, a classifier outputs benign/malicious. Not every point in feature space is the image of a realizable packet stream.

### 2.2 Attack families

- **FGSM** (Goodfellow et al., 2015): single-step sign of the loss gradient; baseline.
- **PGD** (Madry et al., 2018): iterated projected steps; strong first-order white-box attack.
- **HopSkipJump / Boundary / ZOO:** query-based black-box attacks; applicable to non-differentiable models such as random forests.
- **GAN-based evasion:** generate evasive vectors whose distribution resembles benign traffic.

### 2.3 Knowledge

White-box (gradients), grey-box (surrogate/transfer), black-box (queries only). A deployed NIDS is often closest to black-box; white-box remains the worst-case stress test.

### 2.4 Feature space vs. problem space

Feature-space attacks edit the vector. Problem-space attacks edit packets; features change only as a consequence. This protocol is **constrained feature-space**: a projection/filter approximating problem-space legality without yet crafting packets. Valid-sample rate makes residual unrealizability visible. True packet-level validation is future work.

### 2.5 What “realistic” means here

A perturbation must be **functionality-preserving**, **protocol-valid**, and **interdependency-consistent**. Typical rules: immutable protocol/port/flag-type identifiers; non-negative, often increase-only counts; `total_packets = fwd + bwd`; rates consistent with counts and duration.

---

## 3. Related work

He et al. (2023) survey adversarial ML for NIDS and flag the realizability mismatch. Ennaji et al. (2024) catalogue domain constraints and the difficulty of mapping feature-space δ back to valid traffic. Sharma and Chen (2024) document inconsistent evaluation protocols. Alhussien et al. (2024) show that *constraining* attacks changes transferability and defense conclusions. This preprint complements that line by releasing a **mask + harness** and a **paired unconstrained/constrained matrix** on identical splits.

| Attack | Threat model | Space | Respects realistic constraints? | Notes |
| --- | --- | --- | --- | --- |
| FGSM (Goodfellow et al., 2015) | White-box | Feature | No (default) | Single-step baseline |
| PGD (Madry et al., 2018) | White-box | Feature | No (default); mask can be applied | Robustness standard |
| HopSkipJump / Boundary | Black-box (decision) | Feature | No (default) | Works on RF |
| ZOO | Black-box (scores) | Feature | No (default) | Query-expensive |
| GAN-based evasion | Grey/black-box | Often feature; problem-oriented | Partial | Distribution-aware |
| Constrained attacks (Alhussien et al., 2024) | Mixed | Feature + domain constraints | Yes | Closest prior |

**Gap.** Unconstrained feature-space ASR dominates published tables. Missing is a shared, released protocol that reports **valid-sample rate** next to ASR so follow-on papers can cite a common baseline.

---

## 4. Evaluation protocol

### 4.1 Data (intended benchmark)

Use **one** public dataset with identical preprocessing for every cell:

- **CICIDS-2017** (Sharafaldin et al., 2018) — CICFlowMeter CSVs; preferred for constraint provenance.
- **or UNSW-NB15** (Moustafa & Slay, 2015).

Raw files are **not** redistributed. Download from the official providers and place them as in `data/README.md`. Splits: stratified train/val/test with a recorded seed (`config.yaml`: seed 42; val/test 0.15 each). Standardization is fit on train only.

### 4.2 Models

- **MLP** (PyTorch, ART-wrapped) — white-box FGSM/PGD.
- **Random forest** (scikit-learn) — black-box HopSkipJump/GAN only (no gradients).

### 4.3 Attacks and defense

Implemented with the **Adversarial Robustness Toolbox** (FGSM, PGD, HopSkipJump, optional ZOO) plus a simple GAN attack. Each attack is run **unconstrained** and **constrained** (mask projection + validity filter). Defense: **PGD adversarial training** (Madry et al., 2018), then re-evaluate the matrix.

### 4.4 Constraint mask (central artifact)

For `x' = x + δ`, require:

- **Mutability:** immutable features have `δ_i = 0` (protocol, many ports/flags).
- **Direction:** e.g. increase-only packet/byte counts.
- **Range:** non-negativity and protocol boxes.
- **Interdependencies:** e.g. total packets = forward + backward; ratios repaired after projection.

The implementation (`src/constraints.py`) projects then scores **valid-sample rate**. Dataset-specific column maps are marked `TODO` until the export is frozen.

### 4.5 Metrics and matrix

- **ASR** — fraction of originally detected malicious samples that become benign after perturbation.
- **Mean L2 / L∞** of `δ`.
- **Valid-sample rate** — fraction of adversarial samples satisfying the mask (unconstrained cells expose unrealizable “successes”).

Matrix: `{FGSM, PGD, HopSkipJump, GAN} × {unconstrained, constrained} × {no defense, adv. training} × {MLP, RF (black-box attacks only)}`.

---

## 5. Experimental setup

**Software (pipeline-validation run, 2026-07-31):** Python 3.13.3, PyTorch 2.10.0+cpu, ART 1.20.1, scikit-learn 1.8.0, Windows 10, CPU. Unit tests: **27 passed** (`pytest tests/ -v`).

**Reproduction:**

```text
pip install -r requirements.txt
python -m pytest tests/ -v
python experiments/run_experiment.py --synthetic --model mlp --attack all
# Real data (after placing CSVs under data/raw/):
python experiments/run_experiment.py --model mlp --attack all
```

---

## 6. Results

### 6.1 Pipeline validation (synthetic data — not a NIDS finding)

Gaussian-blob synthetic features, MLP, full 16-cell matrix (`RESULTS.md`, artifact `results/mlp_all_20260731-195454`). Clean test accuracy ≈ 0.993.

**Undefended MLP**

| Attack | Regime | ASR | Valid-rate | Mean L2 |
| --- | --- | ---: | ---: | ---: |
| FGSM | unconstrained | 0.000 | 0.327 | 0.346 |
| FGSM | constrained | 0.000 | 1.000 | 0.480 |
| PGD | unconstrained | 0.000 | 0.327 | 0.346 |
| PGD | constrained | 0.000 | 1.000 | 0.480 |
| HopSkipJump | unconstrained | **1.000** | **0.000** | 2.923 |
| HopSkipJump | constrained | **0.027** | **1.000** | 2.325 |
| GAN | unconstrained | 0.000 | 0.380 | 0.133 |
| GAN | constrained | 0.000 | 1.000 | 0.325 |

**What this shows (mechanics only).** Unconstrained HopSkipJump “succeeds” on every attacked malicious sample **and none of those samples are valid under the mask**. Constraining forces valid-rate = 1 and ASR falls to 0.027. Gradient attacks did not evade this easy synthetic MLP at the configured ε (ASR = 0); that is a limitation of the smoke-test distribution, not evidence that FGSM/PGD are weak on CICIDS. RF black-box cells were not completed in that run.

### 6.2 Intended CICIDS-2017 / UNSW-NB15 matrix (pending)

| Attack | Model | ASR (unconst.) | ASR (const.) | Valid-rate (unconst.) | L2 / L∞ |
| --- | --- | --- | --- | --- | --- |
| FGSM | MLP | TBD | TBD | TBD | TBD |
| PGD | MLP | TBD | TBD | TBD | TBD |
| HopSkipJump | MLP | TBD | TBD | TBD | TBD |
| HopSkipJump | RF | TBD | TBD | TBD | TBD |
| GAN | MLP | TBD | TBD | TBD | TBD |
| GAN | RF | TBD | TBD | TBD | TBD |

Defense-effect and ranking-change tables: **TBD** after a real-dataset run. Predicted pattern (to confirm or refute): constrained ASR ≪ unconstrained ASR; unconstrained valid-rate low; ranking may reorder.

---

## 7. Discussion

If the CICIDS/UNSW gap matches the synthetic *mechanism* (high unconstrained ASR, near-zero valid-rate), then many published NIDS ASR figures are not operational threats. Defenders should demand **valid-sample rate** next to ASR and should measure adversarial training under the same mask the network actually imposes. This preprint’s job is to freeze that measurement so later papers cite one protocol.

---

## 8. Limitations

Single-dataset scope once chosen; mask may be incomplete or overly strict; constrained feature-space ≠ packet crafting; synthetic Section 6.1 does not transfer; attack/defense coverage is three families plus one defense; RF matrix incomplete.

---

## 9. Ethics

Evasion research is framed **defensively**: improve evaluation of detectors. No exploit payloads. Datasets remain with their providers. Do not treat synthetic ASR as evidence against production NIDS.

---

## 10. Conclusion and next versions

We release a constraint-aware adversarial-NIDS **protocol and harness**, and we document a synthetic run that exercises valid-sample accounting. **v0.2** of this preprint should replace Section 6.2 with CICIDS-2017 or UNSW-NB15 measurements, then serve as the arXiv record cited by the AI-SEC 2027 workshop submission (target deadline 10 December 2026).

---

## Reproducibility

| Item | Location |
| --- | --- |
| Code | `https://github.com/sathya-ksamy-nsu/constraint-aware-nids` |
| Config / seed | `config.yaml` (`seed: 42`) |
| Mask | `src/constraints.py` |
| Pipeline-validation log | `RESULTS.md` |
| Data download | `data/README.md` (do not redistribute raw CIC/UNSW files) |

---

## Acknowledgments

Comments from colleagues and workshop reviewers will be added in later versions.

---

## References

He, K., Kim, D. S., & Asghar, M. R. (2023). Adversarial machine learning for network intrusion detection systems: A comprehensive survey. *IEEE Communications Surveys & Tutorials, 25*(1), 538–566. https://doi.org/10.1109/COMST.2022.3233793

Alhussien, N., Aleroud, A., Melhem, A., & Khamaiseh, S. Y. (2024). Constraining adversarial attacks on network intrusion detection systems: Transferability and defense analysis. *IEEE Transactions on Network and Service Management, 21*(3), 2751–2772. https://doi.org/10.1109/TNSM.2024.3357316

Sharma, S., & Chen, Z. (2024). A systematic study of adversarial attacks against network intrusion detection systems. *Electronics, 13*(24), 5030. https://doi.org/10.3390/electronics13245030

Ennaji, S., De Gaspari, F., Hitaj, D., Bidi, A. K., & Mancini, L. V. (2024). *Adversarial challenges in network intrusion detection systems: Research insights and future prospects* (arXiv:2409.18736). https://arxiv.org/abs/2409.18736

Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*. https://arxiv.org/abs/1706.06083

Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR*. https://arxiv.org/abs/1412.6572

Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP*.

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. *Military Communications and Information Systems Conference (MilCIS)*.
