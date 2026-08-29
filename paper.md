# How Realistic Are "Successful" Evasion Attacks? A Constraint-Aware Reproducible Comparison of Adversarial Attacks Against ML-Based Network Intrusion Detection

**Author:** Sathyaraj Kolandasamy

**Institution:** Nova Southeastern University — College of Computing, AI, and Cybersecurity

**Dual-purpose paper submitted for:**

| Course | Title | Instructor |
| --- | --- | --- |
| CISC 670 | Artificial Intelligence | Dr. Wei Li |
| ISEC 660 | Advanced Network Security | Dr. Wei Li |

**Date:** 2026-07-30

> **ArXiv preprint (Phase 1):** see [`arxiv-preprint.md`](arxiv-preprint.md) and [`PREPRINT.md`](PREPRINT.md). This file remains the long working draft (including course framing). Submit `arxiv-preprint.md` to arXiv, not this file.

> **Draft status.** This is a full working draft of the paper. The methodology, threat model, related-work synthesis, and experiment design are complete, but **final experimental results are pending**. All cells and passages that depend on experiments that have not yet been run are explicitly marked **TBD**. Numeric findings in the Results section are stated as *anticipated* outcomes derived from the hypothesis and prior literature — they are **not** measured results and must be replaced with real measurements before submission.

---

## Abstract

Machine-learning-based network intrusion detection systems (NIDS) are increasingly deployed to detect novel and evolving attacks, yet a large and growing body of adversarial-machine-learning research reports high "attack success rates" (ASR) against them. Recent 2023–2024 surveys warn that many of these reported successes are **unrealistic**: they optimize perturbations directly in feature space, modifying quantities that an attacker cannot freely control in live traffic and thereby violating protocol semantics and feature-interdependency constraints. This paper tests a narrow, falsifiable hypothesis — that **constraining perturbations to semantically valid, functionality-preserving changes substantially lowers reported attack success rates and reorders the ranking of attacks and defenses** relative to unconstrained perturbations. Using a standard benchmark dataset (CICIDS-2017 or UNSW-NB15) with identical preprocessing and data splits, we compare three concrete attacks spanning distinct threat models — Projected Gradient Descent (white-box, gradient-based), a decision/score-based black-box attack (HopSkipJump or ZOO), and a GAN-based generative attack — against the same target models (a multilayer perceptron and a random forest), with and without an adversarial-training defense, implemented through the Adversarial Robustness Toolbox (ART). Each attack is evaluated twice, unconstrained and under a formally specified constraint mask, and we report ASR, L2/L∞ perturbation size, and a valid-sample rate. The central contribution is a **reusable, reproducible, constraint-aware evaluation protocol** — a realistic-perturbation mask, an evaluation harness, and fixed benchmark splits — that follow-on work can cite and extend. Experimental results are pending and reported as TBD. (~215 words)

---

## 1. Introduction

### 1.1 Motivation: the rise of ML-based NIDS

Modern enterprise, cloud, and carrier networks generate traffic at volumes and with a diversity that overwhelm traditional signature-based intrusion detection. Signature approaches match observed traffic against a database of known-malicious patterns; they are precise for previously catalogued threats but structurally blind to novel or polymorphic attacks for which no signature yet exists. To close this gap, the field has increasingly turned to machine-learning-based network intrusion detection systems (NIDS), which learn statistical regularities of benign and malicious traffic from labeled flow records and generalize to patterns not seen verbatim during training. Models ranging from multilayer perceptrons (MLPs) and tree ensembles such as random forests to deep convolutional and recurrent architectures are now common in both the research literature and production security stacks (He et al., 2023). Because these detectors increasingly sit on the critical path of security decision-making, their trustworthiness is a first-order security concern rather than an academic curiosity.

### 1.2 The adversarial evasion threat

Adversarial machine learning has repeatedly demonstrated that small, deliberately crafted perturbations to an input can flip a classifier's decision while remaining nearly imperceptible under the metric the analyst happens to be using (Goodfellow et al., 2015; Madry et al., 2018). Applied to a NIDS, this phenomenon is not a benign misclassification: a successful *evasion* means that malicious traffic is confidently labeled benign and passes through undetected — a direct, exploitable security failure. The adversarial-NIDS literature has consequently exploded, cataloguing gradient-based, decision-based, score-based, and generative attacks that report evasion success rates frequently exceeding 90% against undefended models (He et al., 2023; Sharma & Chen, 2024).

### 1.3 The "unrealistic evaluation" gap

A recurring and increasingly criticized flaw underlies many of these headline numbers. Most attacks are formulated and evaluated entirely in **feature space**: the adversary is granted the ability to nudge each numerical feature of a flow record independently and continuously, subject only to a norm budget. But a NIDS feature vector is a *derived summary* of an underlying packet stream — it encodes counts, durations, byte totals, flag tallies, and inter-arrival statistics that are mechanically linked to one another and to the protocol that produced them. An attacker who can only send packets cannot set "mean forward inter-arrival time" to an arbitrary value while leaving "flow duration" and "total forward packets" untouched, because those quantities are computed from the same packets. Feature-space perturbations therefore routinely (i) violate **protocol constraints** (impossible header/flag combinations), (ii) break **feature interdependencies** (mutually inconsistent counts, durations, and byte totals), and (iii) modify features that are **not attacker-controllable** or whose modification would **destroy the functionality of the attack itself** (Alhussien et al., 2024; Ennaji et al., 2024). Recent surveys and systematic studies name this gap explicitly and call for realistic, functionality-preserving, and reproducible evaluation (He et al., 2023; Sharma & Chen, 2024; Ennaji et al., 2024). The practical consequence is that reported ASR values may be inflated, defenses may be validated against threats that cannot physically occur, and the community lacks a shared, reproducible way to measure *realistic* robustness.

### 1.4 Research question and hypothesis

This paper asks a single, focused question:

> **Research question.** When adversarial perturbations against an ML-based NIDS are restricted to semantically valid, functionality-preserving changes, how do reported attack success rates and the relative ranking of attacks and defenses change compared with unconstrained perturbations?

We commit to a narrow, falsifiable hypothesis:

> **Hypothesis.** Constraining perturbations to semantically valid, functionality-preserving changes **lowers reported attack success rates** and **reorders the attack/defense ranking** relative to unconstrained perturbations.

This hypothesis yields three directly testable predictions: (1) constrained ASR is substantially and significantly lower than unconstrained ASR for each attack; (2) the ranking of attacks by effectiveness under the constraint mask differs from the unconstrained ranking, so that the strongest unconstrained attack is not necessarily the strongest realistic attack; and (3) the measured benefit of the adversarial-training defense changes when attacks are evaluated under realistic constraints. The hypothesis is falsified if applying the constraint mask leaves ASR values and rankings essentially unchanged.

### 1.5 Contributions

This work makes the following contributions:

1. **A formal, reusable constraint specification** — a "realistic-perturbation" mask for adversarial-NIDS evaluation that encodes, per feature, mutability, direction and range limits, and interdependency relationships that any perturbation must preserve to correspond to sendable traffic.
2. **A reproducible, constraint-aware evaluation harness and benchmark splits** — fixed preprocessing, fixed train/validation/test partitions with recorded seeds, and one-command reproduction, so that unconstrained and constrained regimes are compared on identical data.
3. **An empirical, constraint-aware comparison** of three attacks (PGD, a decision/score-based attack, and a GAN-based attack) across two target models, with and without adversarial training, quantifying how ASR, perturbation size, and the valid-sample rate shift under realistic constraints (results TBD).
4. **A synthesis and comparison table of prior attacks** annotated with whether each respects realistic constraints, making explicit the evaluation gap this work addresses.

The remainder of the paper is organized as follows. Section 2 develops the background and threat model. Section 3 synthesizes related work and states the gap. Section 4 specifies the methodology, including the constraint mask. Section 5 describes the experimental setup and reproducibility protocol. Section 6 presents placeholder result tables and anticipated findings. Sections 7–9 provide discussion, limitations, and conclusions, and Section 10 is a reproducibility appendix.

---

## 2. Background and Threat Model

### 2.1 The ML-based NIDS pipeline

A typical ML-based NIDS processes traffic through a fixed pipeline. Raw packets are first grouped into **flows** (bidirectional sequences sharing a five-tuple of source/destination IP, source/destination port, and protocol). A **feature-extraction** stage (for example, the CICFlowMeter tool used to build CICIDS-2017) summarizes each flow into a fixed-length numerical vector: packet and byte counts in each direction, flow duration, inter-arrival-time statistics, TCP flag counts, header lengths, and various rates and ratios. These vectors are **preprocessed** (missing-value handling, encoding, normalization/standardization) and passed to a trained **classifier** that outputs a benign/malicious label (or per-class scores). The learned decision boundary in this feature space is the object that adversarial attacks target. Crucially, the mapping from packets to features is many-to-one and constrained: not every point in the numerical feature space is the image of some realizable packet stream.

### 2.2 Adversarial examples and attack families

An adversarial example is an input `x' = x + δ` crafted so that the model assigns it a different (attacker-desired) label than `x`, typically while keeping the perturbation `δ` "small" under some norm. The attacks relevant to this study span four families:

- **FGSM (Fast Gradient Sign Method).** The seminal single-step gradient attack (Goodfellow et al., 2015): `x' = x + ε · sign(∇ₓ L(x, y))`. It moves every feature by a fixed magnitude ε in the direction that most increases the loss. FGSM is fast, weak, and serves as a baseline anchor.
- **PGD (Projected Gradient Descent).** The canonical strong first-order white-box attack (Madry et al., 2018): it iterates FGSM-style steps, projecting back into an ε-ball after each step. PGD is widely regarded as a reliable measure of worst-case first-order robustness and is the primary gradient-based attack in this study.
- **Decision- and score-based black-box attacks.** When the adversary lacks gradients, it can query the model and use the returned label (decision-based) or confidence scores (score-based) to estimate a successful perturbation. **ZOO** (Zeroth-Order Optimization) estimates gradients from score queries via finite differences; **HopSkipJump** and the **Boundary** attack are decision-based, walking along the decision boundary using only hard labels. These are directly relevant to non-differentiable targets such as random forests.
- **GAN-based attacks.** A generative adversarial network can be trained to synthesize adversarial or evasive feature vectors (or perturbations) whose distribution mimics benign traffic while retaining malicious intent, offering a generative, distribution-aware alternative to per-sample optimization.

### 2.3 White-, grey-, and black-box threat models

Adversarial threat models are distinguished by the adversary's knowledge of the target:

- **White-box.** Full knowledge — architecture, parameters, and gradients. Enables PGD/FGSM. Represents a worst-case, most-capable adversary and is useful for stress-testing.
- **Grey-box.** Partial knowledge — e.g., the model family, feature set, or training data, but not exact parameters. Common in transfer-attack settings where a surrogate model is trained and its adversarial examples are transferred to the target (Alhussien et al., 2024).
- **Black-box.** Query-only access. The adversary observes outputs (labels or scores) for chosen inputs and must craft evasions without internal access. Decision/score-based attacks operate here; this is often the most realistic setting for an external attacker facing a deployed NIDS.

### 2.4 Feature-space vs. problem-space attacks

A central distinction, emphasized by the recent surveys, is between **feature-space** and **problem-space** attacks. A *feature-space* attack manipulates the numerical feature vector directly, ignoring whether a corresponding packet stream exists. A *problem-space* attack manipulates the actual object under the attacker's control — here, the packets or the traffic-generating behavior — and lets the feature vector change only as a downstream consequence. Problem-space attacks are inherently constrained: the attacker can add padding packets, delay transmissions, or split flows, but cannot, for example, reduce the number of packets already sent or set two mechanically linked features to inconsistent values. The gap between feature-space optimism and problem-space reality is precisely what inflates reported ASR, and it is the phenomenon this paper is designed to quantify.

### 2.5 Realistic, functionality-preserving, protocol-valid perturbations

For a perturbation to correspond to an attack an adversary could actually launch, it must be **functionality-preserving** (the traffic still accomplishes the malicious goal), **protocol-valid** (header fields, flag combinations, and value ranges remain legal), and **consistent with feature interdependencies** (derived quantities remain mutually coherent). Examples of interdependency constraints in flow features include: total packets equals the sum of forward and backward packets; flow duration must be non-negative and consistent with inter-arrival times; byte counts must be non-negative and bounded by packet counts times the maximum segment size; and rate features (bytes/second, packets/second) are deterministic functions of counts and duration. Some features are effectively **immutable** from the attacker's side (for instance, features derived from the victim's responses, or destination-side counters), while others are mutable only in one direction (an attacker can usually *add* packets or *increase* delay far more easily than remove or shrink them). Formalizing these relationships is the purpose of the constraint mask defined in Section 4.4.

---

## 3. Related Work

### 3.1 Surveys framing the gap

He et al. (2023) provide a comprehensive survey of adversarial machine learning for NIDS, taxonomizing attacks and defenses and drawing explicit attention to the mismatch between feature-space attack evaluation and the realizability of those attacks in network traffic. Their survey is the anchor reference establishing both the breadth of reported attacks and the community's awareness that evaluation realism is an unresolved problem. Ennaji et al. (2024) extend this framing, cataloguing the specific adversarial challenges in NIDS — including domain constraints, feature interdependencies, and the difficulty of mapping feature-space perturbations back to valid traffic — and outlining research directions that call for constraint-aware, reproducible evaluation. Sharma and Chen (2024) contribute a systematic study of adversarial attacks against NIDS that consolidates experimental findings across attacks and datasets and again highlights inconsistent, often unrealistic, evaluation protocols as a barrier to comparability.

### 3.2 Constraining attacks and transferability

Alhussien et al. (2024) are most directly aligned with this paper's thesis. They study *constraining* adversarial attacks on NIDS, examining how respecting domain constraints affects attack transferability and defense analysis. Their work provides evidence that constraint-respecting perturbations behave differently from unconstrained ones and that transferability and defense conclusions can change when realism is enforced. This paper complements theirs by packaging the constraint enforcement as a **reusable, released mask and harness** and by conducting a controlled unconstrained-vs-constrained comparison across attack families and a defense on identical splits.

### 3.3 Seminal anchors

The methodological foundations are Goodfellow et al. (2015), who introduced FGSM and the "linear explanation" of adversarial examples, and Madry et al. (2018), who framed adversarial robustness as a min–max optimization problem and established PGD as the standard strong first-order attack and adversarial training as a principled defense. These anchors define the gradient-based attack and the defense used in this study; the NIDS-specific surveys situate them in the network-security domain.

### 3.4 Comparison of prior attacks

The following table synthesizes representative prior attacks along the dimensions most relevant to the evaluation-realism question. It is the technical-comparison core for the ISEC 660 framing. Reported ASR entries are summarized qualitatively from the cited literature; where a specific numeric range is characteristic it is given, otherwise it is noted as reported-high, and gaps are marked TBD to be filled during the literature review.

| Attack | Threat model | Feature- vs. problem-space | Respects realistic constraints? | Reported ASR | Notes |
| --- | --- | --- | --- | --- | --- |
| FGSM (Goodfellow et al., 2015) | White-box | Feature-space | No (default) | Reported high on undefended models | Single-step baseline anchor; weak but fast |
| PGD (Madry et al., 2018) | White-box | Feature-space | No (default); configurable via mask | Reported very high on undefended models | Strong first-order attack; robustness standard |
| ZOO (score-based) | Black-box (scores) | Feature-space | No (default) | Reported high; query-expensive | Finite-difference gradient estimation |
| HopSkipJump / Boundary | Black-box (decision) | Feature-space | No (default) | Reported high; query-efficient decision-based | Applicable to non-differentiable models (RF) |
| GAN-based evasion | Grey/black-box | Problem-space-oriented | Partially (by design) | Reported high; varies by design | Learns distribution of evasive/benign traffic |
| Constrained attacks (Alhussien et al., 2024) | White/grey/black-box | Feature-space with domain constraints | Yes | Lower than unconstrained; transferability shifts | Direct evidence constraints change outcomes |

### 3.5 Identified gap

Across these works two facts recur: (a) unconstrained feature-space attacks dominate the reported literature and produce headline ASR values, and (b) the surveys and the constraint-focused study agree that these values may not reflect realizable threats. What is missing is a **shared, reproducible, constraint-aware protocol** — a released constraint specification plus an evaluation harness and fixed splits — that lets any researcher measure the *realistic* robustness of a NIDS and compare attacks and defenses on equal footing. This paper supplies exactly that artifact and uses it to test whether enforcing realism lowers ASR and reorders rankings.

---

## 4. Methodology

### 4.1 Datasets, preprocessing, and splits

We use one standard, widely cited NIDS benchmark — **CICIDS-2017** or **UNSW-NB15** — and document the rationale for the final choice (dataset selection is an open decision carried from the research notes; CICIDS-2017 offers CICFlowMeter-derived features with clear packet-level provenance, which aids constraint specification, while UNSW-NB15 offers a well-curated mix of modern attack categories). Whichever is selected, **all attacks and defenses use identical preprocessing and splits**: a fixed feature encoding, removal or imputation of malformed/infinite entries, standardization fit on the training split only, and fixed train/validation/test partitions. Every random operation (splitting, shuffling, model initialization, attack sampling) uses recorded seeds so that all configurations see exactly the same data. The preprocessing pipeline and split indices are released as part of the reproducibility artifact.

### 4.2 Target models and training

Two target models are trained on identical splits to span the differentiable/non-differentiable divide:

- **Multilayer perceptron (MLP).** A feedforward network (e.g., two-to-three hidden layers with ReLU activations) trained with cross-entropy. Being differentiable, it enables white-box gradient attacks (FGSM, PGD) directly.
- **Random forest (RF).** A tree ensemble that is non-differentiable, motivating black-box/decision-based attacks (HopSkipJump, ZOO) and transfer attacks from a surrogate.

Both models are trained to a competitive clean accuracy that is recorded as a baseline (clean accuracy TBD). Hyperparameters, training epochs/estimators, and early-stopping criteria are logged in the appendix.

### 4.3 Attacks compared

Three attacks spanning distinct threat models are implemented via the **Adversarial Robustness Toolbox (ART)**, with FGSM retained as a single-step anchor:

1. **PGD** — white-box, gradient-based, against the MLP.
2. **A decision/score-based black-box attack** — **HopSkipJump** (decision-based) and/or **ZOO** (score-based) — against both models, and the natural choice for the RF.
3. **A GAN-based attack** — a generative model trained to produce evasive feature vectors, representing a problem-space-oriented, distribution-aware attacker.

Each attack is run in **two regimes**: *unconstrained* (standard feature-space perturbation as commonly reported) and *constrained* (perturbations restricted by the mask of Section 4.4). Using off-the-shelf ART implementations keeps the study reproducible and feasible.

### 4.4 The constraint mask

The **constraint mask** is the central artifact of this work. It is a formal specification, defined once per dataset, that partitions and bounds the feature set so that any perturbation it admits corresponds to a semantically valid, functionality-preserving, protocol-valid sample. Formally, for a feature vector `x ∈ ℝⁿ`, an admissible perturbed sample `x' = x + δ` must satisfy:

- **Mutability partition.** Each feature `i` is labeled **immutable** (`δᵢ = 0`; e.g., victim-side/destination-derived counters, protocol identifiers, features not under attacker control) or **mutable**. Only mutable features may change.
- **Direction constraints.** For mutable features whose realizable manipulation is one-directional, `δᵢ ≥ 0` or `δᵢ ≤ 0` (e.g., an attacker can add forward packets or increase inter-arrival delay but cannot retract packets already sent).
- **Range/box constraints.** Each mutable feature stays within physically legal bounds `[lᵢ, uᵢ]` (non-negativity for counts/durations, protocol-imposed maxima for header lengths and flag counts).
- **Interdependency constraints.** A set of equality/inequality relations must continue to hold after perturbation, e.g., `total_packets = fwd_packets + bwd_packets`; rate features equal their defining ratios (`bytes_per_sec = total_bytes / flow_duration`); byte totals bounded by packet counts and maximum segment size; durations consistent with inter-arrival statistics. Perturbations that break any relation are projected back onto the feasible set or rejected.

Operationally, the mask is applied inside the attack loop (as a projection/clipping step and a validity filter), so attacks optimize *within* the realizable region rather than being filtered only after the fact. The mask is released as a machine-readable **realistic-perturbation specification** so that follow-on work can reuse or extend it. (Concrete per-feature entries for the chosen dataset: TBD — to be finalized when the dataset is fixed.)

### 4.5 Defense: adversarial training

We evaluate a single, standard defense — **adversarial training** (Madry et al., 2018) — in which the target model is retrained on a mixture of clean and adversarial examples (generated with PGD). All attacks are then re-run against the hardened model, in both regimes, so that the defense's measured benefit can be compared under unconstrained versus constrained evaluation.

### 4.6 Metrics and experiment matrix

We report:

- **Attack success rate (ASR)** — the fraction of malicious test samples pushed to a benign prediction — reported separately for unconstrained and constrained regimes.
- **Perturbation size** — mean **L2** and **L∞** norms of `δ`, quantifying how large a change each attack requires.
- **Valid-sample rate** — the fraction of adversarial samples that satisfy the constraint mask; for unconstrained attacks this exposes how often "successful" evasions are in fact unrealizable.

The full experiment matrix is the cross-product below.

| Dimension | Settings |
| --- | --- |
| Perturbation regime | Unconstrained · Constrained (mask) |
| Defense | No defense · Adversarial training |
| Attacks | PGD · HopSkipJump/ZOO · GAN-based (+ FGSM anchor) |
| Target models | MLP · Random Forest |
| Datasets | CICIDS-2017 **or** UNSW-NB15 |

Full cross-product = attacks × models × {unconstrained, constrained} × {defense, no-defense}. ASR is reported with confidence intervals over multiple seeds, and unconstrained-vs-constrained differences are assessed with an appropriate paired significance test (or bootstrap CIs) to confirm they are not attributable to random variation.

---

## 5. Experimental Setup

### 5.1 Hardware and software

Experiments are implemented in **Python**, using **scikit-learn** for the random forest and preprocessing, **PyTorch** for the MLP and the GAN, and the **Adversarial Robustness Toolbox (ART)** for attack and defense implementations and evaluation. Exact package versions are pinned in a lockfile and reported in the appendix (versions TBD). Hardware (CPU/GPU model, RAM) is recorded for runtime reproducibility (hardware spec TBD).

### 5.2 Seeds and reproducibility

All sources of randomness — data splitting and shuffling, model initialization, minibatch ordering, attack sampling and initialization, and GAN training — are seeded with recorded, fixed values. Splits are stored as explicit index files rather than regenerated on the fly, so that every configuration and every rerun operates on byte-identical data. A single driver script reproduces the full matrix end to end.

### 5.3 Status of results

**All quantitative results in this draft are placeholders (TBD).** The pipeline, models, attacks, mask, and metrics are specified and ready to run; the numbers reported in Section 6 are *anticipated* outcomes implied by the hypothesis and prior literature, clearly labeled as such, and must be replaced by measured values before submission.

---

## 6. Results

> **Note.** Every numeric cell in this section is **TBD**. Narrative statements describe *anticipated* findings under the hypothesis and are explicitly labeled as anticipated, not measured. Do not cite these as results until the experiments are run.

### 6.1 Clean baseline performance

| Model | Dataset | Clean accuracy | Clean F1 (malicious) |
| --- | --- | --- | --- |
| MLP | CICIDS-2017 / UNSW-NB15 | TBD | TBD |
| Random Forest | CICIDS-2017 / UNSW-NB15 | TBD | TBD |

### 6.2 Attack comparison: unconstrained vs. constrained

| Attack | Model | ASR (unconstrained) | ASR (constrained) | L2 (mean) | L∞ (mean) | Valid-sample rate (unconstrained) |
| --- | --- | --- | --- | --- | --- | --- |
| FGSM (anchor) | MLP | TBD | TBD | TBD | TBD | TBD |
| PGD | MLP | TBD | TBD | TBD | TBD | TBD |
| HopSkipJump / ZOO | MLP | TBD | TBD | TBD | TBD | TBD |
| HopSkipJump / ZOO | Random Forest | TBD | TBD | TBD | TBD | TBD |
| GAN-based | MLP | TBD | TBD | TBD | TBD | TBD |
| GAN-based | Random Forest | TBD | TBD | TBD | TBD | TBD |

### 6.3 Defense effect (adversarial training)

| Attack | Model | ASR no-defense (unconstrained) | ASR + adv. training (unconstrained) | ASR no-defense (constrained) | ASR + adv. training (constrained) |
| --- | --- | --- | --- | --- | --- |
| PGD | MLP | TBD | TBD | TBD | TBD |
| HopSkipJump / ZOO | Random Forest | TBD | TBD | TBD | TBD |
| GAN-based | MLP | TBD | TBD | TBD | TBD |

### 6.4 Attack ranking (unconstrained vs. constrained)

| Rank | Unconstrained (by ASR) | Constrained (by ASR) |
| --- | --- | --- |
| 1 | TBD | TBD |
| 2 | TBD | TBD |
| 3 | TBD | TBD |

### 6.5 Anticipated findings (not measured)

Under the hypothesis, we *anticipate* the following patterns; each must be confirmed or refuted by the completed experiments:

- **Anticipated:** constrained ASR is substantially lower than unconstrained ASR for every attack, because the mask forbids many of the perturbations that produce feature-space "successes." The magnitude of the drop is expected to be largest for unconstrained gradient attacks (PGD, FGSM), which freely exploit immutable and interdependent features.
- **Anticipated:** the **valid-sample rate** of unconstrained attacks is low, quantifying how many reported evasions are unrealizable — the core evidence for the "unrealistic evaluation" critique.
- **Anticipated:** the **ranking reorders** — an attack that is designed to respect problem-space structure (the GAN-based attack) or that is easier to project into the feasible set may outrank PGD under the constraint mask even though PGD dominates unconstrained.
- **Anticipated:** the **measured benefit of adversarial training changes** under realistic constraints; a defense that looks strong against unconstrained attacks may look weaker (or stronger) once attacks are confined to realizable perturbations.

If, contrary to these anticipations, ASR and rankings are essentially unchanged after applying the mask, the hypothesis is refuted and that null result is itself a reportable contribution.

---

## 7. Discussion

### 7.1 What a constrained-vs-unconstrained gap would imply

A large gap between unconstrained and constrained ASR would provide direct, reproducible evidence that a substantial fraction of the adversarial-NIDS literature's reported success is an artifact of evaluating in an unrealizable feature space. It would mean that headline robustness numbers systematically overstate real risk, and that both attackers' capabilities and defenders' anxieties have been calibrated against threats that cannot be launched over the wire. Conversely, a small gap would be reassuring — it would suggest that feature-space evaluation is an acceptable proxy for realizable threats, at least for the studied attacks and dataset.

### 7.2 Ranking changes and their consequences

If the attack ranking reorders under constraints, then choosing which attack to defend against based on unconstrained benchmarks is unsound: defenders could be hardening against the wrong worst case. A constraint-aware benchmark reprioritizes defensive effort toward the attacks that remain effective when realism is enforced. This is the practical payoff of the released mask and harness — it lets a defender rank threats by *realistic* effectiveness rather than by feature-space convenience.

### 7.3 Implications for deploying ML-based NIDS

For practitioners, the study speaks to how much to trust robustness claims when procuring or deploying an ML-based NIDS. It argues for demanding constraint-aware evaluation — evidence that a detector resists *realizable* evasions, not merely feature-space ones — and for reporting the valid-sample rate alongside ASR whenever an attack is claimed to succeed. It also cautions that adversarial training's apparent benefit should be measured under the same realistic constraints that the deployed system will actually face.

### 7.4 Connection to both courses

For **CISC 670 (Artificial Intelligence)**, the paper is a narrow, falsifiable study of adversarial robustness that compares three concrete AI attack paradigms — gradient-based, query-based, and generative — and a principled defense, engaging directly with the min–max view of robustness (Madry et al., 2018) and the geometry of adversarial examples (Goodfellow et al., 2015). For **ISEC 660 (Advanced Network Security)**, it sits squarely in the "recent IDS/IPS advances, including adversarial ML for network IDS" area, and its related-work comparison table plus the constraint specification constitute the technical comparison of approaches the course requires. The unifying thread — that security evaluation must model what an adversary can *actually do* on the network — is exactly the intersection of the two disciplines.

---

## 8. Limitations and Threats to Validity

- **Dataset representativeness.** CICIDS-2017 and UNSW-NB15 are standard but imperfect proxies for live enterprise traffic; both have known labeling and distributional idiosyncrasies. Findings may not transfer to other datasets or to real deployments. Using a single dataset (as scoped) limits external validity; multi-dataset replication is future work.
- **Mask completeness (construct validity).** The constraint mask encodes the interdependencies and protocol rules we identify, but it may be incomplete (missing a real constraint, making attacks look stronger than realistic) or overly strict (forbidding a perturbation that is actually realizable, making attacks look weaker). The mask's fidelity is itself a threat to validity and is released precisely so it can be scrutinized and improved.
- **Attack and hyperparameter coverage.** We study three attack families and one defense; conclusions may not generalize to attacks or defenses not covered. Hyperparameter choices (PGD steps/ε, GAN architecture, forest size) can affect ASR, so we report them and use standard settings, but sensitivity is not exhaustively swept.
- **Feature- vs. problem-space fidelity.** Even a constrained feature-space perturbation is an *approximation* of a true problem-space attack; a fully faithful evaluation would craft actual packets and re-extract features. The valid-sample rate mitigates but does not eliminate this gap.
- **Statistical validity.** Reported differences must be supported by confidence intervals over multiple seeds and an appropriate paired test; single-run differences are not treated as conclusive.

---

## 9. Conclusion and Future Work

This paper set out to test whether the "successful" adversarial evasion attacks reported against ML-based NIDS remain successful once perturbations are restricted to what an attacker could realistically send. It contributes a formal, reusable constraint-mask specification, a reproducible evaluation harness with fixed splits, and a controlled comparison of three attacks and one defense across two target models in unconstrained and constrained regimes. The central hypothesis — that realistic constraints lower ASR and reorder attack/defense rankings — is stated in a falsifiable form with explicit predictions and success/refutation criteria. **Experimental results are pending (TBD);** completing the matrix will either substantiate the "unrealistic evaluation" critique with reproducible evidence or, via a null result, show that feature-space evaluation is an acceptable proxy.

**Future work.** Extend the mask to additional datasets (multi-dataset external validity); implement a true problem-space pipeline that crafts packets and re-extracts features to validate the mask; broaden the attack/defense set (additional black-box attacks, detection-based and certified defenses); and release the constraint spec and harness as a community benchmark with a leaderboard-style results table to accrue and standardize follow-on comparisons.

---

## 10. Reproducibility Appendix

- **Repository.** [repo link placeholder — e.g., https://github.com/<user>/constraint-aware-nids] (TBD).
- **Environment.** Python [version TBD]; pinned dependencies including scikit-learn [version TBD], PyTorch [version TBD], and Adversarial Robustness Toolbox [version TBD], captured in a lockfile (`requirements.txt` / `environment.yml`). Hardware: [CPU/GPU/RAM TBD].
- **Seeds.** All random seeds fixed and recorded (data split, shuffling, model init, attack init, GAN training) — values TBD.
- **Data splits.** Train/validation/test index files released so all configurations use byte-identical data.
- **Constraint specification.** The realistic-perturbation mask published as a machine-readable file enumerating, per feature: mutability, direction, `[l, u]` range, and interdependency relations — contents TBD once the dataset is fixed.
- **How to run.** One-command driver reproduces the full matrix: preprocess → train MLP + RF baselines → run attacks (unconstrained, then constrained) → apply adversarial training → re-run attacks → emit metric tables (ASR, L2/L∞, valid-sample rate) and ranking analysis. Exact command and config files: TBD.

---

## References

> The following APA references are reproduced from the proposal scaffold and `research-notes.md` (Topic 1), preserving any `[verify …]` flags exactly. **At least 5 core references are required** for the target courses; the list below meets that bar. During the literature review, add **2–4 more concrete attack/defense papers** — for example, specific ZOO, HopSkipJump/Boundary, or GAN-based evasion works, or additional constraint-aware NIDS studies — to strengthen the comparison table. **Do not fabricate references** beyond verified sources; if a claim needs support you do not yet have, mark it `[citation needed — locate during lit review]`.

1. He, K., Kim, D. S., & Asghar, M. R. (2023). Adversarial machine learning for network intrusion detection systems: A comprehensive survey. *IEEE Communications Surveys & Tutorials, 25*(1), 538–566. https://doi.org/10.1109/COMST.2022.3233793

2. Alhussien, N., Aleroud, A., Melhem, A., & Khamaiseh, S. Y. (2024). Constraining adversarial attacks on network intrusion detection systems: Transferability and defense analysis. *IEEE Transactions on Network and Service Management, 21*(3), 2751–2772. https://doi.org/10.1109/TNSM.2024.3357316

3. Sharma, S., & Chen, Z. (2024). A systematic study of adversarial attacks against network intrusion detection systems. *Electronics, 13*(24), 5030. https://doi.org/10.3390/electronics13245030

4. Ennaji, S., De Gaspari, F., Hitaj, D., Bidi, A. K., & Mancini, L. V. (2024). *Adversarial challenges in network intrusion detection systems: Research insights and future prospects* (arXiv:2409.18736). arXiv. https://arxiv.org/abs/2409.18736

5. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR. https://arxiv.org/abs/1706.06083 (seminal — PGD)

6. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and harnessing adversarial examples*. ICLR. https://arxiv.org/abs/1412.6572 (seminal — FGSM)
