# NBA sequential foul-call dependence — replication package

Replication code for *Testing Sequential Dependence in NBA Foul Calls Using Play-by-Play Data*.

**Repository:** https://github.com/Mariana2000/nba-foul-sequential-replicate

This repository is a **public replication package**. It contains the pipeline, Python package, and manuscript sources needed to reproduce the paper's tables and figures. It intentionally excludes exploratory notebooks, draft notes, and work-in-progress analyses from the private research repository.

## Quick start

```bash
pip install -r requirements.txt
set PYTHONPATH=src   # Windows
# export PYTHONPATH=src   # macOS/Linux

python scripts/01_download_nba_pbp.py --start-season 2018 --end-season 2025
python scripts/02_clean_pbp.py
python scripts/03_build_foul_events.py
python scripts/08_publication_analysis.py
python scripts/13_jqas_possession_robustness.py
python scripts/09_publication_figures.py
python scripts/10_export_manuscript_tables.py
python scripts/12_export_journal_appendix.py
python scripts/06_robustness_checks.py
python scripts/validate_foul_counts.py
```

Then compile the PDF from `docs/arxiv/` (see `REPLICATION.md`).

## Outputs

- `data/processed/foul_events.csv` — analysis-ready foul-event panel
- `outputs/tables/` — regression, placebo, and robustness CSVs
- `outputs/figures/publication_*.png` — main-text figures
- `docs/tables/table*.tex` — LaTeX tables (Table 1 maintained; Tables 2–7 generated)

## Data

Raw play-by-play is downloaded from public NBA sources by `scripts/01_download_nba_pbp.py`. Generated data and outputs are gitignored; see `.gitignore`.

## Citation

If you use this code, please cite the paper (forthcoming). See `docs/arxiv/references.bib`.

## License

MIT License. See [LICENSE](LICENSE).
