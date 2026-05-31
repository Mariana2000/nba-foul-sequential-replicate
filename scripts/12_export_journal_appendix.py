"""Export journal appendix tables (full coefficients, alternative clustering)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from whistle_balance.config import PROCESSED_DIR, TABLE_DIR
from whistle_balance.modeling import SEQUENTIAL_FORMULA, extended_sample
from whistle_balance.publication import fit_logit_clustered

TERM_LABELS = {
    "foul_diff_home_minus_away_before": "Game foul diff (home $-$ away)",
    "period_foul_diff_home_minus_away_before": "Period foul diff (home $-$ away)",
    "last_foul_against_home": "Last foul on home",
    "time_since_last_foul": "Time since last foul (seconds)",
    "home_in_bonus_before": "Home team in bonus",
    "away_in_bonus_before": "Away team in bonus",
}

MODEL_LABELS = {
    "baseline_cluster": "Baseline logit",
    "sequential_cluster": "Sequential logit",
    "bonus_cluster": "Bonus controls",
    "team_fe_cluster": "Team FE",
}


def export_clustering_comparison(out_csv: Path, out_tex: Path) -> None:
    df = pd.read_csv(PROCESSED_DIR / "foul_events.csv", dtype={"game_id": str})
    sample = extended_sample(df)

    rows: list[dict] = []
    for cluster_col, label in [("game_id", "Cluster by game"), ("home_team", "Cluster by home team")]:
        model = fit_logit_clustered(SEQUENTIAL_FORMULA, sample, cluster_col=cluster_col)
        for term in [
            "foul_diff_home_minus_away_before",
            "period_foul_diff_home_minus_away_before",
            "last_foul_against_home",
        ]:
            rows.append(
                {
                    "cluster": label,
                    "term": term,
                    "coef": float(model.params[term]),
                    "std_err": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "n_obs": int(model.nobs),
                }
            )

    frame = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Alternative standard-error clustering (sequential logit)}",
        r"\label{tab:clustering}",
        r"\small",
        r"\begin{threeparttable}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Clustering & Variable & Coefficient & SE \\",
        r"\midrule",
    ]
    for cluster in ["Cluster by game", "Cluster by home team"]:
        sub = frame[frame["cluster"] == cluster]
        first = True
        for _, row in sub.iterrows():
            label = TERM_LABELS.get(row["term"], row["term"])
            prefix = cluster if first else ""
            first = False
            lines.append(
                f"{prefix} & {label} & {row['coef']:.3f} & ({row['std_err']:.3f}) \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend(
        [
            r"\end{tabular}",
            r"\begin{tablenotes}[flushleft]",
            r"\small",
            r"\item[] Same sequential logit specification as Table~\ref{tab:main_regression}, column~(2).",
            r"\item[] Main text reports game-level clustering; home-team clustering is a robustness check on within-team correlation of residuals.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_full_regression(out_tex: Path) -> None:
    long = pd.read_csv(TABLE_DIR / "publication_clustered_main.csv")
    long = long[long["term"].isin(TERM_LABELS)].copy()
    long["model_label"] = long["model"].map(MODEL_LABELS)
    long["term_label"] = long["term"].map(TERM_LABELS)

    models = [
        "Baseline logit",
        "Sequential logit",
        "Bonus controls",
        "Team FE",
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Full regression output: key coefficients (clustered logit, SE clustered by game)}",
        r"\label{tab:full_regression}",
        r"\small",
        r"\begin{threeparttable}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(models) + "}",
        r"\toprule",
        " & ".join([""] + models) + r" \\",
        r"\midrule",
    ]

    for term_key, term_label in TERM_LABELS.items():
        coef_cells = [term_label]
        se_cells = [""]
        for model_name in MODEL_LABELS:
            sub = long[(long["model"] == model_name) & (long["term"] == term_key)]
            if sub.empty:
                coef_cells.append("")
                se_cells.append("")
            else:
                row = sub.iloc[0]
                coef_cells.append(f"{row['coef']:.3f}")
                se_cells.append(f"({row['std_err']:.3f})")
        lines.append(" & ".join(coef_cells) + r" \\")
        if any(se_cells[1:]):
            lines.append(" & ".join(se_cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\small",
            r"\item[] Dependent variable: indicator that the current foul is called on the home team.",
            r"\item[] All models include score margin, home possession, period fixed effects, seconds remaining, and season fixed effects unless noted.",
            r"\item[] Source: \texttt{outputs/tables/publication\_clustered\_main.csv}.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    docs_tables = Path("docs/tables")
    export_clustering_comparison(
        TABLE_DIR / "robustness_clustering.csv",
        docs_tables / "appendix_clustering.tex",
    )
    export_full_regression(docs_tables / "appendix_full_regression.tex")
    print("Wrote docs/tables/appendix_clustering.tex")
    print("Wrote docs/tables/appendix_full_regression.tex")
    print("Wrote outputs/tables/robustness_clustering.csv")


if __name__ == "__main__":
    main()
