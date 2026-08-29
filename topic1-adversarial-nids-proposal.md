# How Realistic Are "Successful" Evasion Attacks? A Constraint-Aware Reproducible Comparison of Adversarial Attacks Against ML-Based Network Intrusion Detection

**Author:** Sathyaraj Kolandasamy

**Institution:** Nova Southeastern University — College of Computing, AI, and Cybersecurity

**Target courses (dual-purpose paper):**

| Course | Title | Instructor |
| --- | --- | --- |
| CISC 670 | Artificial Intelligence | Dr. Wei Li |
| ISEC 660 | Advanced Network Security | Dr. Wei Li |

**Date:** 2026-07-30

**Binding deadline:** ISEC 660 final paper due **2026-05-03**.

> This document is a self-contained topic proposal that also serves as the scaffold for drafting the full 12+ page paper. All references are reused verbatim from `research-notes.md` (Topic 1); none have been invented. Any `[verify …]` flags from the source are preserved.

---

## 1. Abstract / Summary

Machine-learning-based network intrusion detection systems (NIDS) are increasingly deployed, yet a large body of adversarial-ML research reports high "attack success rates" against them. Recent 2024–2026 surveys warn that many of these successes are **unrealistic**: they perturb features that cannot be freely manipulated in real network traffic, violating protocol semantics and feature-interdependency constraints. This paper tests a narrow, falsifiable hypothesis — that **constraining perturbations to semantically valid, functionality-preserving changes substantially lowers reported attack success rates and reorders the ranking of attacks and defenses**. Using standard datasets (CICIDS-2017 or UNSW-NB15) with identical preprocessing and splits, we compare 2–3 concrete attacks (gradient-based, decision/score-based, and generative) against the same target models, with and without adversarial-training defense, using the Adversarial Robustness Toolbox. The contribution is a **reusable, reproducible constraint-aware evaluation protocol** — a "realistic-perturbation" mask, harness, and benchmark splits — that follow-on work can cite and build upon. (~150 words)

---

## 2. Problem Statement & Motivation

- **Why ML-based NIDS matter.** Modern networks generate volumes and varieties of traffic that overwhelm signature-based detection. ML-based NIDS (MLPs, tree ensembles, deep models) generalize to novel patterns and are now common in both research and production security stacks. Their trustworthiness is therefore a first-order security concern.
- **The evasion threat.** Adversarial machine learning shows that small, deliberately crafted perturbations can flip a classifier's decision. Against a NIDS, a successful evasion means malicious traffic is labeled benign — a direct security failure.
- **Why many reported evasion "successes" are unrealistic.** A recurring flaw in the literature is optimizing perturbations directly in **feature space** without checking whether the resulting feature vector corresponds to any traffic an attacker could actually send. Such perturbations frequently:
  - violate **protocol constraints** (e.g., impossible header/flag combinations),
  - break **feature interdependencies** (e.g., byte counts, packet counts, and durations that must be mutually consistent),
  - modify features that are **not attacker-controllable** or that would **destroy the attack's functionality**.
- **Consequence.** Reported attack success rates (ASR) can be inflated, defenses can be evaluated against threats that cannot occur, and the community lacks a shared, reproducible way to measure *realistic* robustness. The 2024–2026 surveys explicitly name this gap and call for realistic, reproducible evaluation — the motivation for this work.

---

## 3. Research Question & Hypothesis

**Research question.** When adversarial perturbations against an ML-based NIDS are restricted to semantically valid, functionality-preserving changes, how do reported attack success rates and the relative ranking of attacks and defenses change compared with unconstrained perturbations?

**Narrow, falsifiable hypothesis (from `research-notes.md`, Topic 1).**
> Constraining perturbations to **semantically valid, functionality-preserving** changes **lowers reported attack success rates** and **reorders the attack/defense ranking** relative to unconstrained perturbations.

**Directly testable predictions.**
1. Constrained ASR < unconstrained ASR for each attack (substantial, statistically significant drop).
2. The *ranking* of attacks by effectiveness under the constraint mask differs from the unconstrained ranking (i.e., the "best" unconstrained attack is not necessarily the best realistic attack).
3. The measured benefit of the adversarial-training defense changes when evaluated under realistic constraints.

The hypothesis is falsified if constraining perturbations leaves ASR and rankings essentially unchanged.

---

## 4. Approaches to be Compared (the technical core)

We apply 2–3 concrete adversarial attacks spanning distinct threat models to the **same** NIDS problem, plus one defense:

- **PGD (Projected Gradient Descent)** — gradient-based, white-box; the canonical strong first-order attack (Madry et al., 2018). FGSM (Goodfellow et al., 2015) is the seminal single-step anchor.
- **ZOO** or **HopSkipJump / Boundary** — decision/score-based, black-box; estimates the needed information without model gradients.
- **A GAN-based attack** — generative approach to synthesizing adversarial/evasive traffic feature vectors.
- **Defense: adversarial training** — retrain the target model on adversarial examples and re-measure all attacks.

Each attack is evaluated **twice**: unconstrained (feature-space, as commonly reported) and constrained (restricted by the realistic-perturbation mask). The table below is the ISEC 660 technical-comparison core; result cells are placeholders to be filled during experiments.

### Comparison Table (template — fill during experiments)

| Attack | Threat model (white/grey/black-box) | Feature-space vs. problem-space | Respects realistic constraints? | Reported ASR (unconstrained) | ASR (constrained) | Perturbation size (L2 / L∞) |
| --- | --- | --- | --- | --- | --- | --- |
| FGSM (baseline anchor) | White-box | Feature-space | No (default) | TBD | TBD | TBD |
| PGD | White-box | Feature-space → constrained | Configurable via mask | TBD | TBD | TBD |
| ZOO / HopSkipJump / Boundary | Black-box (score/decision) | Feature-space → constrained | Configurable via mask | TBD | TBD | TBD |
| GAN-based attack | Grey/black-box | Problem-space-oriented | Partially (by design) | TBD | TBD | TBD |
| PGD **+ adversarial training** (defense) | White-box vs. hardened model | Feature-space → constrained | Configurable via mask | TBD | TBD | TBD |

---

## 5. Methodology / Research Plan

### 5.1 Datasets, preprocessing, and splits
- **Datasets:** CICIDS-2017 **or** UNSW-NB15 (choose one; document rationale). A dataset decision is an open item carried from `research-notes.md`.
- **Identical preprocessing & splits** across all attacks/defenses: fixed feature encoding, normalization, and train/validation/test partitions, with recorded random seeds so every configuration sees the same data.

### 5.2 Target models
- **MLP** (differentiable — enables white-box gradient attacks) and **Random Forest** (non-differentiable — motivates black-box/transfer attacks). Both trained on identical splits.

### 5.3 Constraint mask (the central artifact)
- Define a **constraint mask** specifying which features are mutable and how, such that any perturbation yields a **semantically valid, functionality-preserving** sample:
  - immutable / non-attacker-controllable features are locked;
  - interdependent features (counts, durations, byte totals) are kept mutually consistent;
  - protocol-invalid combinations are rejected.
- The mask is released as a **reusable "realistic-perturbation" specification**.

### 5.4 Tooling
- **Adversarial Robustness Toolbox (ART)** in Python for attacks, defense, and evaluation; off-the-shelf implementations keep the study reproducible and feasible in 1–2 weeks.

### 5.5 Metrics
- **Attack success rate (ASR)** — unconstrained vs. constrained.
- **Perturbation size** — L2 and L∞ norms.
- **Valid-sample rate** — fraction of adversarial samples satisfying the constraint mask.

### 5.6 Experiment matrix
| Dimension | Settings |
| --- | --- |
| Perturbation regime | Unconstrained · Constrained (mask) |
| Defense | No defense · Adversarial training |
| Attacks | PGD · ZOO/HopSkipJump/Boundary · GAN-based (+ FGSM anchor) |
| Target models | MLP · Random Forest |
| Datasets | CICIDS-2017 **or** UNSW-NB15 |

Full cross-product = attacks × models × {unconstrained, constrained} × {defense, no-defense}.

### 5.7 Rough timeline (fits before 2026-05-03)
- **Weeks 1–2:** finalize dataset; preprocessing pipeline; train MLP + RF baselines; reproduce clean accuracy.
- **Weeks 3–4:** implement/run unconstrained attacks via ART; record baseline ASR/perturbation sizes.
- **Week 5:** define + implement the constraint mask; add valid-sample checking; re-run attacks constrained.
- **Week 6:** adversarial-training defense; re-run full matrix ± defense.
- **Week 7:** statistical analysis; populate comparison + results tables; ranking analysis.
- **Week 8:** write-up, reproducibility appendix, internal review; buffer before the **2026-05-03** deadline.

---

## 6. Dual-Course Mapping

### CISC 670 — Artificial Intelligence (Dr. Wei Li)
- **Frontier AI, narrow and testable:** adversarial robustness of ML models with a single falsifiable hypothesis.
- **Compares 2–3 concrete approaches:** gradient-based (PGD) vs. decision/score-based (ZOO/HopSkipJump/Boundary) vs. generative (GAN), with an explicit pros/cons discussion.
- **Hands-on feasible (Option 2):** implementable in ~1–2 weeks using ART; satisfies the frontier-AI project path if chosen.
- **Not a "survey of surveys":** concrete, testable experimental core.

### ISEC 660 — Advanced Network Security (Dr. Wei Li)
- **Area 3 — recent IDS/IPS advances, including adversarial ML for network IDS.**
- **Technical comparison of 2–3 approaches** is the paper's core, anchored by the Related Work comparison table.
- **Top-venue references** (IEEE COMST, IEEE TNSM, ICLR anchors) meet the ≥ 5 references / 12+ page requirements.

---

## 7. Expected Contributions & Citation Strategy

**Contributions.**
- A **reusable constraint specification** (the "realistic-perturbation" mask) for adversarial-NIDS evaluation.
- An **open evaluation harness + script** and fixed **data splits** enabling one-command reproduction.
- An **empirical, constraint-aware comparison** showing how ASR and rankings shift under realistic constraints.

**Gap prior work ignores (state explicitly in the introduction).** Much prior work reports ASR from feature-space perturbations that are not realizable in real traffic; the field lacks a shared, reproducible protocol for *realistic* robustness. This paper supplies one.

**Citation strategy (from `research-notes.md`).**
- Release **code + data splits + one-command repro**.
- Post an **arXiv (cs.CR) preprint early**.
- Ship the **constraint spec as a reusable artifact** that follow-on work cites.
- Use a **precise, searchable title** and a **clear comparison table + leaderboard-style result**.
- Cite **newest 2024–2026 works** plus **seminal anchors** (Goodfellow et al., 2015; Madry et al., 2018).

---

## 8. Full Paper Outline

1. **Abstract**
2. **Introduction** — the "unrealistic evaluation" gap in adversarial-NIDS research; explicit contributions list.
3. **Background & Threat Model** — white-box / grey-box / black-box adversaries; **feature-space vs. problem-space** attacks; **NIDS feature constraints** (which features are mutable / functionality-preserving).
4. **Related Work** — comparison table of prior attacks noting **whether each respects realistic constraints** (the ISEC 660 technical-comparison core).
5. **Methodology**
   - **5.1 Datasets & identical preprocessing/splits** (CICIDS-2017 or UNSW-NB15).
   - **5.2 Target models** — e.g., MLP + Random Forest.
   - **5.3 Attacks + constraint-mask definition** — PGD, ZOO, Boundary/HopSkipJump, GAN-based; define the valid-perturbation mask.
   - **5.4 Defense = adversarial training; metrics** — ASR, L2 / L∞ perturbation, valid-sample rate.
6. **Experiments & Results** — unconstrained vs. constrained ASR; ranking changes; defense effect; statistical significance.
7. **Discussion** — interpretation; implications for how the field should evaluate robustness.
8. **Limitations & Threats to Validity** — dataset representativeness; mask completeness; attack/hyperparameter coverage.
9. **Conclusion & Future Work**
10. **Reproducibility Appendix** — repo link, environment, seeds, constraint spec, one-command repro.

---

## 9. Evaluation / Success Criteria

The hypothesis is **confirmed** if, across attacks and both target models:

- [ ] Constrained ASR is **substantially and significantly lower** than unconstrained ASR.
- [ ] The **attack ranking reorders** between unconstrained and constrained regimes.
- [ ] The **measured defense benefit changes** under realistic constraints.
- [ ] The **valid-sample rate** cleanly separates realistic from unrealistic adversarial examples.

The hypothesis is **refuted** if ASR and rankings are essentially unchanged after applying the constraint mask.

**Statistical significance note.** Report ASR with confidence intervals over multiple seeds/runs; use an appropriate significance test (e.g., paired test across matched samples, or bootstrap CIs) to confirm that unconstrained-vs.-constrained differences are not attributable to random variation.

---

## 10. Ethics & Reproducibility

- **Isolated experimentation.** All attacks run offline against locally trained models on public datasets — no live networks or third-party systems.
- **Responsible / defensive framing.** The goal is *more honest robustness evaluation*, not enabling real-world evasion; the released artifact strengthens defensive evaluation.
- **Reproducibility appendix (checklist).**
  - [ ] Public repository link (placeholder).
  - [ ] Pinned environment / dependency versions (Python, ART).
  - [ ] Fixed random **seeds** recorded.
  - [ ] **Constraint spec** (realistic-perturbation mask) published.
  - [ ] Fixed **data splits** and one-command reproduction script.
- **NSU generative-AI policy reminder.** Generative-AI use requires **instructor pre-approval**; obtain approval before using any AI assistance on this graded work.

---

## 11. References (APA)

> Reproduced exactly from `research-notes.md` (Topic 1 references), including any `[verify …]` flags. **At least 5 core references are required**; during the literature review, add **2–4 more concrete attack/defense papers** (e.g., specific ZOO, HopSkipJump/Boundary, or GAN-based evasion works) to strengthen the comparison. Do **not** fabricate references beyond verified sources.

1. He, K., Kim, D. S., & Asghar, M. R. (2023). Adversarial machine learning for network intrusion detection systems: A comprehensive survey. *IEEE Communications Surveys & Tutorials, 25*(1), 538–566. https://doi.org/10.1109/COMST.2022.3233793
2. Alhussien, N., Aleroud, A., Melhem, A., & Khamaiseh, S. Y. (2024). Constraining adversarial attacks on network intrusion detection systems: Transferability and defense analysis. *IEEE Transactions on Network and Service Management, 21*(3), 2751–2772. https://doi.org/10.1109/TNSM.2024.3357316
3. Sharma, S., & Chen, Z. (2024). A systematic study of adversarial attacks against network intrusion detection systems. *Electronics, 13*(24), 5030. https://doi.org/10.3390/electronics13245030
4. Ennaji, S., De Gaspari, F., Hitaj, D., Bidi, A. K., & Mancini, L. V. (2024). *Adversarial challenges in network intrusion detection systems: Research insights and future prospects* (arXiv:2409.18736). arXiv. https://arxiv.org/abs/2409.18736
5. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR. https://arxiv.org/abs/1706.06083 (seminal — PGD)
6. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples*. ICLR. https://arxiv.org/abs/1412.6572 (seminal — FGSM)
