# arXiv LaTeX manuscript

Working paper source for *Sequential Foul-Call Dependence in the NBA*.

## Layout

```
docs/arxiv/
  main.tex              # Master document
  preamble.tex          # Packages and macros
  references.bib        # Bibliography (placeholder entries)
  sections/             # Section bodies
  README.md
docs/tables/            # Generated LaTeX tables (Tables 1--6)
outputs/figures/        # Generated PNG figures
```

Markdown drafts (extended prose) live alongside under `docs/section_5_*.md`, `docs/section_6_discussion.md`, `docs/section_7_conclusion.md`.

## Prerequisites

- LaTeX distribution (TeX Live / MiKTeX) with `latexmk`, `pdflatex`, `bibtex`
- Python pipeline outputs already generated (figures + tables)

## Build

From repository root (PowerShell):

```powershell
$env:PYTHONPATH="src"
python scripts/10_export_manuscript_tables.py
python scripts/09_publication_figures.py

cd docs/arxiv
latexmk -pdf -interaction=nonstopmode main.tex
```

Or manually:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `docs/arxiv/main.pdf`

## Before submission

1. Replace author names, affiliations, and `\thanks{...}` emails in `main.tex`.
2. Update GitHub URL in `references.bib` (`nbafoulbalance2026`).
3. Replace placeholder citations (`carpenter2005`) with final bibliography.
4. Confirm figure paths resolve (`../../outputs/figures/` from `docs/arxiv/`).
5. Optional: switch to `\documentclass{article}` + arXiv `.zip` with flattened figure paths.

## Table / figure labels

| Item | Label |
|------|-------|
| Table 1 | `tab:summary` |
| Table 2 | `tab:main_regression` |
| Table 3 | `tab:game_fe` |
| Table 4 | `tab:marginal_effects` |
| Table 5 | `tab:placebo` |
| Table 6 | `tab:heterogeneity` |
| Figure 1 | `fig:main` |
| Figure 2 | `fig:robust` |
| Figure 3 | `fig:hetero` |

## arXiv upload tips

- Upload `main.tex`, `preamble.tex`, `references.bib`, all `sections/*.tex`, all `../tables/table*.tex` (keep relative `\input{../tables/...}` paths or flatten).
- Include figure PNGs or compile PDF locally and upload PDF only.
- Add `\pdfoutput=1` if required by your TeX setup (usually automatic with pdflatex).
