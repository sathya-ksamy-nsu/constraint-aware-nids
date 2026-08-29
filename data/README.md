# Data — obtaining CICIDS-2017 / UNSW-NB15

**Raw datasets are NOT committed to this repository.** They are large and
distributed under their providers' own license terms. Download them yourself,
place the files in the expected layout below, and point `config.yaml` at them.

> Reproducibility note: only the *preprocessing pipeline*, *split logic*, and
> *fixed seeds* live in this repo (see `src/data_loader.py`). The raw bytes stay
> on your machine so that every run reconstructs byte-identical splits from the
> same source files.

---

## Option A — CICIDS-2017 (recommended for constraint work)

- **Provider:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick.
- **Official page:** https://www.unb.ca/cic/datasets/ids-2017.html
- **What to download:** the machine-learning CSVs produced by CICFlowMeter
  (the `MachineLearningCSV` / `GeneratedLabelledFlows` archives). These contain
  ~80 flow features per record plus a `Label` column.
- **License / terms:** free for research use; UNB requires that you **cite** the
  dataset paper (Sharafaldin, Lashkari & Ghorbani, 2018) and abide by the terms
  on the official page. Review and comply before use.
- **Why it aids constraint specification:** CICIDS-2017 features are derived by
  CICFlowMeter with clear packet-level provenance, which makes it easier to
  reason about which features are attacker-controllable and how they interrelate.

### Expected layout
```
data/raw/cicids2017/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  ...                                  (all provided CSVs; loader concatenates *.csv)
```
Set in `config.yaml`:
```yaml
dataset:
  name: "cicids2017"
  raw_dir: "data/raw/cicids2017"
  file_glob: "*.csv"
  label_column: "Label"
```

---

## Option B — UNSW-NB15

- **Provider:** Cyber Range Lab, UNSW Canberra.
- **Official page:** https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **What to download:** the partitioned CSVs
  `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` (or the four
  full CSV parts). Features include `proto`, `service`, `state`, packet/byte
  counts, durations, and a binary `label` (plus `attack_cat`).
- **License / terms:** free for research use with **citation** of the dataset
  papers (Moustafa & Slay, 2015). Review and comply with the page's terms.

### Expected layout
```
data/raw/unsw-nb15/
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```
Set in `config.yaml`:
```yaml
dataset:
  name: "unsw-nb15"
  raw_dir: "data/raw/unsw-nb15"
  file_glob: "*.csv"
  label_column: "label"
  benign_labels: ["0", "normal", "Normal"]
```

---

## Constraint-mask assumptions (domain notes)

The constraint mask (`src/constraints.py`, paper Section 4.4) encodes these
domain assumptions. **These are defaults that MUST be reviewed against the exact
column names of whichever dataset export you download** — search for `TODO` in
`src/constraints.py`.

- **Immutable / not attacker-controllable** (perturbation forced to zero):
  - protocol identifier (`Protocol` / `proto`),
  - service port (`Destination Port` / `dst_port`),
  - TCP flag-type indicator columns (`... Flag Count`, `service`, `state`),
  - any victim-/destination-side derived counters.
  Rationale: an attacker cannot relabel the protocol or change which flag *type*
  a feature counts without changing the attack itself.
- **Non-negativity:** packet counts, byte counts, durations, and lengths are
  `>= 0`.
- **Direction (add-only):** count/packet/byte/duration features are treated as
  `INCREASE_ONLY` by default — an attacker can pad packets or add delay but
  cannot retract traffic already sent.
- **Interdependencies:** derived features are kept consistent, e.g.
  `Total Packets = Total Fwd Packets + Total Backward Packets`; rate features
  equal their defining ratios (`bytes/sec = total_bytes / flow_duration`); byte
  totals are bounded by `packets × MSS`.
- **Integer features:** packet/flag counts are rounded to integers after
  projection.

### TODOs the student must complete
- [ ] Confirm the exact `label_column` and `benign_labels` for your export.
- [ ] Replace the keyword heuristics in `build_default_spec()` with an explicit
      per-column table (exact immutable columns and exact `[lo, hi]` ranges) for
      the chosen dataset.
- [ ] Add the concrete interdependency relations using the dataset's real column
      names (the generic `sum_relation` / `ratio_relation` / `upper_bound_relation`
      helpers are ready to wire up).
- [ ] Record dataset citations in `paper.md` References.
