"""JQAS possession-cycle and related robustness checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

from whistle_balance.config import PROCESSED_DIR, TABLE_DIR
from whistle_balance.modeling import SEQUENTIAL_FORMULA, extended_sample
POSSESSION_SEQUENTIAL_FORMULA = (
    "foul_against_possession_team ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_possession_team + time_since_last_foul "
    "+ score_margin_home_before "
    "+ C(period) + seconds_remaining_game + C(season)"
)

HOME_SEQUENTIAL_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + time_since_last_foul "
    "+ score_margin_home_before "
    "+ C(period) + seconds_remaining_game + C(season)"
)

SUBSAMPLE_SPECS = [
    ("full_main_sample", "Main sample (excl. offensive)", None),
    ("shooting_fouls_only", "Shooting fouls only", "shooting_foul == 1"),
    ("home_possession", "Home possession only", "home_possession == 1"),
    ("away_possession", "Away possession only", "home_possession == 0"),
]

TRACKED = [
    ("last_foul_against_home", "Last foul on home"),
    ("last_foul_against_possession_team", "Last foul on possession team"),
    ("foul_diff_home_minus_away_before", "Game foul diff (home $-$ away)"),
]


def _add_possession_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    poss = out["home_possession"].astype(float)
    out["foul_against_possession_team"] = np.where(
        poss == 1,
        out["foul_against_home"],
        1 - out["foul_against_home"],
    )
    out["last_foul_against_possession_team"] = out.groupby("game_id")[
        "foul_against_possession_team"
    ].shift(1)
    return out


def _pick_home_formula(sample: pd.DataFrame) -> str:
    if sample["home_possession"].nunique(dropna=True) <= 1:
        return HOME_SEQUENTIAL_FORMULA
    return SEQUENTIAL_FORMULA


def _fit_logit_robust(formula: str, data: pd.DataFrame):
    model = smf.logit(formula=formula, data=data)
    try:
        return model.fit(
            disp=False,
            maxiter=200,
            method="newton",
            cov_type="cluster",
            cov_kwds={"groups": data["game_id"]},
        )
    except np.linalg.LinAlgError:
        return model.fit(
            disp=False,
            maxiter=200,
            method="bfgs",
            cov_type="HC1",
        )


def run_possession_robustness(data: pd.DataFrame) -> pd.DataFrame:
    base = _add_possession_outcomes(extended_sample(data))
    rows: list[dict] = []

    for spec_id, label, query in SUBSAMPLE_SPECS:
        sample = base if not query else base.query(query)
        sample = sample.dropna(
            subset=[
                "foul_against_possession_team",
                "last_foul_against_possession_team",
                "foul_diff_home_minus_away_before",
                "period_foul_diff_home_minus_away_before",
            ]
        )
        if len(sample) < 500:
            rows.append({"spec": spec_id, "label": label, "n_obs": len(sample), "status": "insufficient"})
            continue

        home_model = _fit_logit_robust(_pick_home_formula(sample), sample)
        poss_sample = sample.dropna(subset=["last_foul_against_possession_team"])
        poss_model = _fit_logit_robust(POSSESSION_SEQUENTIAL_FORMULA, poss_sample)

        for term, term_label in TRACKED:
            if term == "last_foul_against_possession_team":
                model = poss_model
                n_obs = int(poss_model.nobs)
            else:
                model = home_model
                n_obs = int(home_model.nobs)
            if term not in model.params.index:
                continue
            rows.append(
                {
                    "spec": spec_id,
                    "label": label,
                    "term": term,
                    "term_label": term_label,
                    "coef": float(model.params[term]),
                    "std_err": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "n_obs": n_obs,
                    "status": "ok",
                }
            )

    return pd.DataFrame(rows)


def export_possession_appendix_tex(frame: pd.DataFrame, out: Path) -> None:
    ok = frame[frame["status"] == "ok"]
    specs = ok.drop_duplicates(["spec", "label"])[["spec", "label", "n_obs"]]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Possession-cycle robustness (sequential logit, clustered by game)}",
        r"\label{tab:possession_robustness}",
        r"\small",
        r"\begin{threeparttable}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Subsample & $N$ & Last foul on home & Last foul on poss.\ team & Game foul diff \\",
        r"\midrule",
    ]

    for _, spec in specs.iterrows():
        sub = ok[ok["spec"] == spec["spec"]]
        cells = [spec["label"], f"{int(spec['n_obs']):,}"]
        for term in [
            "last_foul_against_home",
            "last_foul_against_possession_team",
            "foul_diff_home_minus_away_before",
        ]:
            row = sub[sub["term"] == term]
            if row.empty:
                cells.append("--")
            else:
                r = row.iloc[0]
                cells.append(f"{r['coef']:.3f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\small",
            r"\item[] Main sample excludes technical, flagrant, and offensive fouls.",
            r"\item[] ``Last foul on possession team'' uses outcome \texttt{foul\_against\_possession\_team} and lagged possession-team indicator; other columns use the home-team outcome specification.",
            r"\item[] Tests whether previous-call reversal survives shooting-foul subsamples and possession states.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    path = PROCESSED_DIR / "foul_events.csv"
    df = pd.read_csv(path, dtype={"game_id": str})
    frame = run_possession_robustness(df)
    csv_out = TABLE_DIR / "jqas_possession_robustness.csv"
    frame.to_csv(csv_out, index=False)
    export_possession_appendix_tex(frame, Path("docs/tables/table7_possession_robustness.tex"))
    print("Wrote jqas possession robustness outputs")


if __name__ == "__main__":
    main()
