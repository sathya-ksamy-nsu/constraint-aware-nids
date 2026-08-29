# Submit this paper to arXiv

**Do not upload Markdown as the article.** Upload **LaTeX source** (preferred) and/or a **PDF**.

## Files to upload

| File | Upload? |
| --- | --- |
| `latex/main.tex` | **Yes** — this is the paper |
| Compiled `main.pdf` | Yes, if you have it |
| This GitHub URL | In the PDF (already there) and optionally in Comments |
| `arxiv-preprint.md` | No |
| `src/`, datasets, `.venv` | No |

## arXiv form fields

- **Title:** How Realistic Are "Successful" Evasion Attacks? A Constraint-Aware Reproducible Comparison of Adversarial Attacks Against ML-Based Network Intrusion Detection
- **Authors:** Sathyaraj Kolandasamy
- **Affiliation:** Nova Southeastern University
- **Abstract:** copy the `abstract` environment from `latex/main.tex`
- **Comments:** Protocol and open evaluation harness. v0.1: synthetic pipeline validation only; CICIDS-2017/UNSW-NB15 results forthcoming. Code: https://github.com/sathya-ksamy-nsu/constraint-aware-nids
- **Primary category:** `cs.CR`
- **Cross-list:** `cs.LG`
- **License:** arXiv default or CC BY 4.0 for the *text*; code is MIT.

## Compile a PDF locally (if you have TeX)

```text
cd latex
pdflatex main.tex
pdflatex main.tex
```

Or Overleaf: New project → upload `latex/main.tex` → Download PDF.

## Dual-submission

Check the AI-SEC 2027 / Springer CFP before posting if the workshop PDF will be substantially the same. arXiv-first is common in this community; still confirm the 2027 CFP.
