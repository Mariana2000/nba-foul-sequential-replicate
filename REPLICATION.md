# Replication guide

This document mirrors Appendix A of the manuscript.

## Environment

```bash
pip install -r requirements.txt
```

Set `PYTHONPATH=src` from the repository root.

## Pipeline (in order)

1. `scripts/01_download_nba_pbp.py`
2. `scripts/02_clean_pbp.py`
3. `scripts/03_build_foul_events.py`
4. `scripts/08_publication_analysis.py`
5. `scripts/13_jqas_possession_robustness.py`
6. `scripts/09_publication_figures.py`
7. `scripts/10_export_manuscript_tables.py`
8. `scripts/12_export_journal_appendix.py`
9. `scripts/06_robustness_checks.py`
10. `scripts/validate_foul_counts.py`

## Manuscript

```bash
cd docs/arxiv
latexmk -pdf main.tex
```

## Random seeds

Placebo draws use seeds 42–141 (100 draws). See `src/whistle_balance/publication.py`.

## Excluded from this package

Exploratory scripts (`04_descriptive_analysis.py`, `05_model_foul_direction.py`, `07_l2m_extension.py`), notebooks, draft markdown under `docs/`, and submission checklists from the private working repository.
