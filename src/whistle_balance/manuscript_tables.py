"""Export publication manuscript tables from analysis CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import numpy as np

from whistle_balance.config import TABLE_DIR

ROW_SPECS = [
    ("foul_diff_home_minus_away_before", "Game foul diff (home − away)"),
    ("period_foul_diff_home_minus_away_before", "Period foul diff (home − away)"),
    ("last_foul_against_home", "Last foul on home"),
    ("time_since_last_foul", "Time since last foul (seconds)"),
    ("home_in_bonus_before", "Home team in bonus"),
    ("away_in_bonus_before", "Away team in bonus"),
]

MODEL_SPECS = [
    ("baseline_cluster", "(1) Baseline"),
    ("sequential_cluster", "(2) Sequential"),
    ("bonus_cluster", "(3) Bonus"),
    ("team_fe_cluster", "(4) Team FE"),
    ("game_fe_lpm", "(5) Game FE LPM"),
]

TABLE3_MODEL_SPECS = [
    ("sequential_cluster", "(1) Sequential logit"),
    ("game_fe_lpm", "(2) Game FE LPM"),
]

AME_MODEL_SPECS = [
    ("baseline_cluster", "(1) Baseline"),
    ("sequential_cluster", "(2) Sequential"),
    ("bonus_cluster", "(3) Bonus"),
    ("team_fe_cluster", "(4) Team FE"),
]

AME_ROW_SPECS = [
    ("foul_diff_home_minus_away_before", "Game foul diff (home − away)"),
    ("period_foul_diff_home_minus_away_before", "Period foul diff (home − away)"),
    ("last_foul_against_home", "Last foul on home"),
    ("time_since_last_foul", "Time since last foul (seconds)"),
    ("home_in_bonus_before", "Home team in bonus"),
    ("away_in_bonus_before", "Away team in bonus"),
]


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _fmt_cell(coef: float, se: float, p_value: float) -> tuple[str, str]:
    if abs(coef) < 0.01 and coef != 0:
        coef_str = f"{coef:.4f}{_stars(p_value)}"
        se_str = f"({se:.4f})"
    else:
        coef_str = f"{coef:.3f}{_stars(p_value)}"
        se_str = f"({se:.3f})"
    return coef_str, se_str


def _latex_tablenotes(notes: list[str]) -> list[str]:
    lines = [r"\begin{tablenotes}[flushleft]", r"\small"]
    for note in notes:
        lines.append(rf"\item[] {note}")
    lines.append(r"\end{tablenotes}")
    return lines


def _load_regression_long() -> pd.DataFrame:
    clustered = pd.read_csv(TABLE_DIR / "publication_clustered_main.csv")
    game_fe = pd.read_csv(TABLE_DIR / "publication_game_fe_lpm.csv")
    game_fe["model"] = "game_fe_lpm"
    long = pd.concat([clustered, game_fe], ignore_index=True)
    return long.set_index(["model", "term"])


def _build_regression_table_frame(
    model_specs: list[tuple[str, str]],
    row_specs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    long = _load_regression_long()
    rows: list[dict] = []
    specs = row_specs or ROW_SPECS
    for term, label in specs:
        row: dict[str, str] = {"Variable": label, "term": term}
        for model, _ in model_specs:
            if (model, term) not in long.index:
                row[model] = ""
                row[f"{model}_se"] = ""
                continue
            rec = long.loc[(model, term)]
            coef_str, se_str = _fmt_cell(float(rec["coef"]), float(rec["std_err"]), float(rec["p_value"]))
            row[model] = coef_str
            row[f"{model}_se"] = se_str
        rows.append(row)

    meta_rows = [
        ("Observations", "n_obs"),
        ("Games", "n_games"),
        ("Pseudo $R^2$ / Within $R^2$", "pseudo_r2"),
    ]
    for label, field in meta_rows:
        row = {"Variable": label, "term": field}
        for model, _ in model_specs:
            sub = long.reset_index()
            sub = sub[sub["model"] == model]
            if sub.empty:
                row[model] = ""
                row[f"{model}_se"] = ""
                continue
            val = sub.iloc[0][field]
            if field == "pseudo_r2":
                row[model] = f"{float(val):.3f}"
            else:
                row[model] = f"{int(val):,}"
            row[f"{model}_se"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def load_table2_frame() -> pd.DataFrame:
    return _build_regression_table_frame(MODEL_SPECS)


def load_table3_frame() -> pd.DataFrame:
    game_fe_rows = [
        ("foul_diff_home_minus_away_before", "Game foul diff (home − away)"),
        ("period_foul_diff_home_minus_away_before", "Period foul diff (home − away)"),
        ("last_foul_against_home", "Last foul on home"),
        ("time_since_last_foul", "Time since last foul (seconds)"),
    ]
    return _build_regression_table_frame(TABLE3_MODEL_SPECS, game_fe_rows)


def load_table4_frame() -> pd.DataFrame:
    ame = pd.read_csv(TABLE_DIR / "publication_marginal_effects.csv")
    ame = ame.set_index(["model", "term"])
    discrete = pd.read_csv(TABLE_DIR / "publication_discrete_changes.csv")
    pred = pd.read_csv(TABLE_DIR / "publication_predicted_probabilities.csv")

    rows: list[dict] = []
    for term, label in AME_ROW_SPECS:
        row: dict[str, str] = {"Variable": label, "term": term}
        for model, _ in AME_MODEL_SPECS:
            if (model, term) not in ame.index:
                row[model] = ""
                row[f"{model}_se"] = ""
                continue
            rec = ame.loc[(model, term)]
            dy = float(rec["dy_dx"])
            se = float(rec["std_err"])
            p = float(rec["p_value"])
            if abs(dy) < 0.01 and dy != 0:
                row[model] = f"{dy:.4f}{_stars(p)}"
                row[f"{model}_se"] = f"({se:.4f})"
            elif abs(se) < 0.001 and se != 0:
                row[model] = f"{dy:.3f}{_stars(p)}"
                row[f"{model}_se"] = f"({se:.4f})"
            else:
                row[model] = f"{dy:.3f}{_stars(p)}"
                row[f"{model}_se"] = f"({se:.3f})"
        rows.append(row)

    last_away = float(pred.loc[(pred["foul_diff_home_minus_away_before"] == 0) & (pred["last_foul_against_home"] == 0), "predicted_p_foul_against_home"].iloc[0])
    last_home = float(pred.loc[(pred["foul_diff_home_minus_away_before"] == 0) & (pred["last_foul_against_home"] == 1), "predicted_p_foul_against_home"].iloc[0])
    disc_away = float(discrete.loc[discrete["last_foul_against_home"] == 0, "discrete_change"].iloc[0])
    disc_home = float(discrete.loc[discrete["last_foul_against_home"] == 1, "discrete_change"].iloc[0])

    rows.append(
        {
            "Variable": "Discrete change: game foul diff +1 (prev. foul on away)",
            "term": "discrete_away",
            "sequential_cluster": f"{disc_away:.3f}",
            "sequential_cluster_se": "",
            "baseline_cluster": "",
            "baseline_cluster_se": "",
            "bonus_cluster": "",
            "bonus_cluster_se": "",
            "team_fe_cluster": "",
            "team_fe_cluster_se": "",
        }
    )
    rows.append(
        {
            "Variable": "Discrete change: game foul diff +1 (prev. foul on home)",
            "term": "discrete_home",
            "sequential_cluster": f"{disc_home:.3f}",
            "sequential_cluster_se": "",
            "baseline_cluster": "",
            "baseline_cluster_se": "",
            "bonus_cluster": "",
            "bonus_cluster_se": "",
            "team_fe_cluster": "",
            "team_fe_cluster_se": "",
        }
    )
    rows.append(
        {
            "Variable": "Predicted P(home foul): prev. foul on away (foul diff = 0)",
            "term": "pred_last_away",
            "sequential_cluster": f"{last_away:.3f}",
            "sequential_cluster_se": "",
            "baseline_cluster": "",
            "baseline_cluster_se": "",
            "bonus_cluster": "",
            "bonus_cluster_se": "",
            "team_fe_cluster": "",
            "team_fe_cluster_se": "",
        }
    )
    rows.append(
        {
            "Variable": "Predicted P(home foul): prev. foul on home (foul diff = 0)",
            "term": "pred_last_home",
            "sequential_cluster": f"{last_home:.3f}",
            "sequential_cluster_se": "",
            "baseline_cluster": "",
            "baseline_cluster_se": "",
            "bonus_cluster": "",
            "bonus_cluster_se": "",
            "team_fe_cluster": "",
            "team_fe_cluster_se": "",
        }
    )
    rows.append(
        {
            "Variable": "Implied shift from previous-call flip (at foul diff = 0)",
            "term": "pred_last_shift",
            "sequential_cluster": f"{last_home - last_away:.3f}",
            "sequential_cluster_se": "",
            "baseline_cluster": "",
            "baseline_cluster_se": "",
            "bonus_cluster": "",
            "bonus_cluster_se": "",
            "team_fe_cluster": "",
            "team_fe_cluster_se": "",
        }
    )
    return pd.DataFrame(rows)


def _export_word_csv(frame: pd.DataFrame, model_cols: list[str], out: Path) -> Path:
    export_rows: list[dict] = []
    for _, row in frame.iterrows():
        coef_row = {"Variable": row["Variable"]}
        se_row = {"Variable": ""}
        for model in model_cols:
            coef_row[model] = row.get(model, "")
            se_row[model] = row.get(f"{model}_se", "")
        export_rows.extend([coef_row, se_row])
    pd.DataFrame(export_rows).to_csv(out, index=False)
    return out


def _export_latex_table(
    frame: pd.DataFrame,
    model_specs: list[tuple[str, str]],
    *,
    out: Path,
    caption: str,
    label: str,
    notes: list[str],
    resize: bool = False,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    model_cols = [m for m, _ in model_specs]
    col_headers = [label for _, label in model_specs]
    ncols = len(model_cols) + 1

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
    ]
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend(
        [
            r"\begin{tabular}{l" + "c" * len(model_cols) + "}",
            r"\toprule",
            " & ".join([""] + col_headers) + r" \\",
            r"\midrule",
        ]
    )

    for _, row in frame.iterrows():
        label_text = str(row["Variable"]).replace("−", r"$-$")
        coef_cells = [label_text]
        se_cells = [""]
        for model in model_cols:
            coef_cells.append(str(row.get(model, "")))
            se_cells.append(str(row.get(f"{model}_se", "")))
        lines.append(" & ".join(coef_cells) + r" \\")
        if any(se_cells[1:]):
            lines.append(" & ".join(se_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.extend(
        [
            r"\end{tabular}",
        ]
    )
    if resize:
        lines.append(r"}")
    if notes:
        lines.extend(_latex_tablenotes(notes))
    lines.extend(
        [
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _export_markdown_table(
    frame: pd.DataFrame,
    model_specs: list[tuple[str, str]],
    *,
    out: Path,
    title: str,
    notes: str,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    model_cols = [m for m, _ in model_specs]
    headers = ["Variable"] + [label for _, label in model_specs]
    md_lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        coef = [row["Variable"]] + [str(row.get(m, "")) for m in model_cols]
        se = [""] + [str(row.get(f"{m}_se", "")) for m in model_cols]
        md_lines.append("| " + " | ".join(coef) + " |")
        if any(se[1:]):
            md_lines.append("| " + " | ".join(se) + " |")
    md_lines.extend(["", notes, ""])
    out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out


def table3_word_csv(path: Path | None = None) -> Path:
    frame = load_table3_frame()
    model_cols = [m for m, _ in TABLE3_MODEL_SPECS]
    return _export_word_csv(frame, model_cols, path or TABLE_DIR / "table3_game_fe_word.csv")


def table3_latex(path: Path | None = None) -> Path:
    frame = load_table3_frame()
    return _export_latex_table(
        frame,
        TABLE3_MODEL_SPECS,
        out=path or Path("docs/tables/table3_game_fe.tex"),
        caption="Within-game identification: sequential logit versus game fixed effects",
        label="tab:game_fe",
        notes=[
            r"Dependent variable: indicator that the current foul is called on the home team.",
            r"Column (1) is the sequential clustered logit from Table 2. Column (2) is a linear probability model with game fixed effects; coefficients are within-game partial effects.",
            r"Both specifications include score margin, home possession, period indicators, and seconds remaining; the logit also includes season fixed effects.",
            r"Standard errors in parentheses, clustered by game. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
        ],
    )


def table3_markdown(path: Path | None = None) -> Path:
    frame = load_table3_frame()
    return _export_markdown_table(
        frame,
        TABLE3_MODEL_SPECS,
        out=path or Path("docs/tables/table3_game_fe.md"),
        title="Table 3. Within-game identification (sequential logit vs game FE LPM)",
        notes="**Notes:** See LaTeX table for full notes.",
    )


def table4_word_csv(path: Path | None = None) -> Path:
    frame = load_table4_frame()
    model_cols = [m for m, _ in AME_MODEL_SPECS]
    return _export_word_csv(frame, model_cols, path or TABLE_DIR / "table4_marginal_effects_word.csv")


def table4_latex(path: Path | None = None) -> Path:
    frame = load_table4_frame()
    return _export_latex_table(
        frame,
        AME_MODEL_SPECS,
        out=path or Path("docs/tables/table4_marginal_effects.tex"),
        caption="Average marginal effects and predicted-probability shifts (sequential logit)",
        label="tab:marginal_effects",
        notes=[
            r"Columns (1)--(4) report average marginal effects (AME) from clustered logistic models; effects are changes in $P(\text{foul against home})$.",
            r"AMEs for continuous regressors are per one-unit increases; the effect for ``Last foul on home'' is the change when the indicator switches from 0 to 1.",
            r"Bottom rows report nonlinear summaries from the sequential model only: discrete changes when game foul diff increases by 1, selected predicted probabilities, and the implied shift from a previous-call flip at foul diff $=0$.",
            r"Standard errors in parentheses for AME rows, clustered by game. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
        ],
        resize=True,
    )


def table4_markdown(path: Path | None = None) -> Path:
    frame = load_table4_frame()
    return _export_markdown_table(
        frame,
        AME_MODEL_SPECS,
        out=path or Path("docs/tables/table4_marginal_effects.md"),
        title="Table 4. Average marginal effects and probability shifts",
        notes="**Notes:** AME = average marginal effect on P(foul against home).",
    )


PLACEbo_ROW_SPECS = [
    ("coef_foul_diff", "Game foul diff (home − away)"),
    ("coef_period_diff", "Period foul diff (home − away)"),
    ("coef_last_foul", "Last foul on home"),
]


def load_table5_frame() -> pd.DataFrame:
    draws = pd.read_csv(TABLE_DIR / "publication_placebo_draws.csv")
    actual = draws.loc[draws["draw"] == "actual"].iloc[0]
    placebo = draws.loc[draws["draw"] != "actual"]

    clustered = pd.read_csv(TABLE_DIR / "publication_clustered_main.csv")
    seq = clustered[clustered["model"] == "sequential_cluster"].set_index("term")

    rows: list[dict] = []
    for col, label in PLACEbo_ROW_SPECS:
        actual_val = float(actual[col])
        pbo = placebo[col].astype(float)
        term = {
            "coef_foul_diff": "foul_diff_home_minus_away_before",
            "coef_period_diff": "period_foul_diff_home_minus_away_before",
            "coef_last_foul": "last_foul_against_home",
        }[col]
        se = float(seq.loc[term, "std_err"]) if term in seq.index else float("nan")

        if col == "coef_last_foul":
            emp_p = float((pbo <= actual_val).mean())
        else:
            emp_p = float("nan")

        rows.append(
            {
                "Variable": label,
                "term": col,
                "actual": f"{actual_val:.3f}",
                "actual_se": f"({se:.3f})",
                "placebo_mean": f"{pbo.mean():.3f}",
                "placebo_sd": f"{pbo.std():.3f}",
                "placebo_min": f"{pbo.min():.3f}",
                "placebo_max": f"{pbo.max():.3f}",
                "empirical_p": "" if np.isnan(emp_p) else f"{emp_p:.2f}",
            }
        )
    rows.append(
        {
            "Variable": "Number of placebo draws",
            "term": "n_draws",
            "actual": "",
            "actual_se": "",
            "placebo_mean": str(len(placebo)),
            "placebo_sd": "",
            "placebo_min": "",
            "placebo_max": "",
            "empirical_p": "",
        }
    )
    return pd.DataFrame(rows)


def table5_word_csv(path: Path | None = None) -> Path:
    frame = load_table5_frame()
    out = path or TABLE_DIR / "table5_placebo_word.csv"
    export = frame[
        [
            "Variable",
            "actual",
            "actual_se",
            "placebo_mean",
            "placebo_sd",
            "placebo_min",
            "placebo_max",
            "empirical_p",
        ]
    ].copy()
    export.columns = [
        "Variable",
        "Actual estimate",
        "Actual SE",
        "Placebo mean",
        "Placebo SD",
        "Placebo min",
        "Placebo max",
        "Empirical p-value",
    ]
    export.to_csv(out, index=False)
    return out


def table5_latex(path: Path | None = None) -> Path:
    frame = load_table5_frame()
    out = path or Path("docs/tables/table5_placebo.tex")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Placebo test: within-game shuffled foul order}",
        r"\label{tab:placebo}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r" & Actual & Placebo & Placebo & Placebo & Placebo & Empirical \\",
        r" & estimate & mean & SD & min & max & $p$-value \\",
        r"\midrule",
    ]

    for _, row in frame.iterrows():
        if row["term"] == "n_draws":
            lines.append(
                f"Number of placebo draws & \\multicolumn{{6}}{{c}}{{{row['placebo_mean']}}} \\\\"
            )
            continue
        label = str(row["Variable"]).replace("−", r"$-$")
        emp = row["empirical_p"] if row["empirical_p"] else "--"
        lines.append(
            " & ".join(
                [
                    label,
                    row["actual"],
                    row["placebo_mean"],
                    row["placebo_sd"],
                    row["placebo_min"],
                    row["placebo_max"],
                    emp,
                ]
            )
            + r" \\"
        )
        if row["actual_se"]:
            lines.append(
                " & ".join(["", row["actual_se"], "", "", "", "", ""]) + r" \\"
            )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\small",
            r"\item[] Actual estimates are from the sequential clustered logit in Table 2, column (2).",
            r"\item[] Placebo samples shuffle foul order within each game, recompute sequential state variables, and re-estimate the same model without clustered standard errors.",
            r"\item[] For ``Last foul on home,'' the empirical $p$-value is the share of placebo draws with coefficients at or below the actual estimate (one-sided).",
            r"\item[] Standard error in parentheses below the actual estimate only.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def table5_markdown(path: Path | None = None) -> Path:
    frame = load_table5_frame()
    out = path or Path("docs/tables/table5_placebo.md")
    headers = [
        "Variable",
        "Actual",
        "Placebo mean",
        "Placebo SD",
        "Placebo min",
        "Placebo max",
        "Empirical p",
    ]
    md_lines = [
        "# Table 5. Placebo test (shuffled within-game foul order)",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["Variable"]),
                    str(row["actual"]),
                    str(row["placebo_mean"]),
                    str(row["placebo_sd"]),
                    str(row["placebo_min"]),
                    str(row["placebo_max"]),
                    str(row["empirical_p"] or "--"),
                ]
            )
            + " |"
        )
        if row["actual_se"] and row["term"] != "n_draws":
            md_lines.append(
                "|  | "
                + row["actual_se"]
                + " |  |  |  |  |  |"
            )
    md_lines.extend(["", "**Notes:** See LaTeX table for full notes.", ""])
    out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out


FOUL_TYPE_LABELS = {
    "shooting": "Shooting fouls",
    "offensive": "Offensive fouls",
    "loose_ball": "Loose-ball fouls",
    "personal": "Personal fouls",
}

INTERACTION_ROW_SPECS = [
    ("foul_diff_home_minus_away_before:C(playoffs)[T.1]", "Game foul diff $\\times$ playoffs"),
    ("foul_diff_home_minus_away_before:close_game", "Game foul diff $\\times$ close game ($|margin|\\leq 10$)"),
    ("last_foul_against_home:foul_diff_home_minus_away_before", "Last foul on home $\\times$ game foul diff"),
]


def load_table6_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    foul_types = pd.read_csv(TABLE_DIR / "publication_foul_type_models.csv")
    foul_types = foul_types[foul_types["status"] == "ok"].copy()

    panel_a: list[dict] = []
    for _, row in foul_types.iterrows():
        label = FOUL_TYPE_LABELS.get(row["foul_type"], str(row["foul_type"]))
        panel_a.append(
            {
                "foul_type": label,
                "n_obs": f"{int(row['n_obs']):,}",
                "coef_foul_diff": f"{row['coef_foul_diff']:.3f}{_stars(float(row['p_foul_diff']))}",
                "coef_period_diff": f"{row['coef_period_diff']:.3f}{_stars(float(row['p_period_diff']))}",
                "coef_last_foul": f"{row['coef_last_foul']:.3f}{_stars(float(row['p_last_foul']))}",
            }
        )

    interactions = pd.read_csv(TABLE_DIR / "publication_interactions.csv")
    panel_b: list[dict] = []
    for term_key, label in INTERACTION_ROW_SPECS:
        if term_key not in interactions["term"].values:
            continue
        rec = interactions.loc[interactions["term"] == term_key].iloc[0]
        coef_str, se_str = _fmt_cell(float(rec["coef"]), float(rec["std_err"]), float(rec["p_value"]))
        panel_b.append(
            {
                "term": label,
                "coef": coef_str,
                "se": se_str,
            }
        )

    return pd.DataFrame(panel_a), pd.DataFrame(panel_b)


def table6_word_csv(path: Path | None = None) -> Path:
    panel_a, panel_b = load_table6_frame()
    out = path or TABLE_DIR / "table6_heterogeneity_word.csv"
    rows: list[dict] = []
    rows.append({"Panel": "A: Foul type subsamples", "Detail": "", "N": "", "Game diff": "", "Period diff": "", "Last foul": ""})
    for _, row in panel_a.iterrows():
        rows.append(
            {
                "Panel": row["foul_type"],
                "Detail": "",
                "N": row["n_obs"],
                "Game diff": row["coef_foul_diff"],
                "Period diff": row["coef_period_diff"],
                "Last foul": row["coef_last_foul"],
            }
        )
    rows.append({"Panel": "B: Interaction terms", "Detail": "", "N": "", "Game diff": "", "Period diff": "", "Last foul": ""})
    for _, row in panel_b.iterrows():
        rows.append(
            {
                "Panel": row["term"],
                "Detail": row["coef"],
                "N": row["se"],
                "Game diff": "",
                "Period diff": "",
                "Last foul": "",
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def table6_latex(path: Path | None = None) -> Path:
    panel_a, panel_b = load_table6_frame()
    out = path or Path("docs/tables/table6_heterogeneity.tex")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Heterogeneity by foul type and selected interactions}",
        r"\label{tab:heterogeneity}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\multicolumn{5}{l}{\textbf{Panel A: Sequential logit by foul type}} \\",
        r"\toprule",
        r"Foul type & $N$ & Game foul diff & Period foul diff & Last foul on home \\",
        r"\midrule",
    ]
    for _, row in panel_a.iterrows():
        lines.append(
            f"{row['foul_type']} & {row['n_obs']} & {row['coef_foul_diff']} & {row['coef_period_diff']} & {row['coef_last_foul']} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{5}{l}{\textbf{Panel B: Interaction terms (full sequential model with interactions)}} \\",
            r"\midrule",
            r"Interaction term & \multicolumn{2}{c}{Coefficient} & \multicolumn{2}{c}{} \\",
            r"\midrule",
        ]
    )
    for _, row in panel_b.iterrows():
        lines.append(f"{row['term']} & \\multicolumn{{2}}{{c}}{{{row['coef']}}} & \\multicolumn{{2}}{{c}}{{{row['se']}}} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\begin{tablenotes}[flushleft]",
            r"\small",
            r"\item[] Panel A re-estimates the sequential logit on foul-type subsamples (clustered SE; stars from $p$-values).",
            r"\item[] Panel B reports interaction coefficients from a model that also includes the main effects in Table 2. Close game indicates $|\text{margin}|\leq 10$ points.",
            r"\item[] Personal-foul coefficients may reflect subset composition; offensive fouls follow different assignment rules (see text).",
            r"\item[] $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def table6_markdown(path: Path | None = None) -> Path:
    panel_a, panel_b = load_table6_frame()
    out = path or Path("docs/tables/table6_heterogeneity.md")
    md = [
        "# Table 6. Heterogeneity by foul type and interactions",
        "",
        "## Panel A: Foul type subsamples",
        "",
        "| Foul type | N | Game foul diff | Period foul diff | Last foul on home |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for _, row in panel_a.iterrows():
        md.append(
            f"| {row['foul_type']} | {row['n_obs']} | {row['coef_foul_diff']} | "
            f"{row['coef_period_diff']} | {row['coef_last_foul']} |"
        )
    md.extend(
        [
            "",
            "## Panel B: Interaction terms",
            "",
            "| Interaction | Coef | SE |",
            "| --- | --- | --- |",
        ]
    )
    for _, row in panel_b.iterrows():
        md.append(f"| {row['term']} | {row['coef']} | {row['se']} |")
    md.extend(["", "**Notes:** See LaTeX table for full notes.", ""])
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


def table2_word_csv(path: Path | None = None) -> Path:
    frame = load_table2_frame()
    out = path or TABLE_DIR / "table2_main_regression_word.csv"
    export_rows: list[dict] = []
    model_cols = [m for m, _ in MODEL_SPECS]
    for _, row in frame.iterrows():
        coef_row = {"Variable": row["Variable"]}
        se_row = {"Variable": ""}
        for model in model_cols:
            coef_row[model] = row.get(model, "")
            se_row[model] = row.get(f"{model}_se", "")
        export_rows.extend([coef_row, se_row])
    pd.DataFrame(export_rows).to_csv(out, index=False)
    return out


def table2_latex(path: Path | None = None) -> Path:
    frame = load_table2_frame()
    out = path or Path("docs/tables/table2_main_regression.tex")
    out.parent.mkdir(parents=True, exist_ok=True)

    model_cols = [m for m, _ in MODEL_SPECS]
    col_headers = [label for _, label in MODEL_SPECS]

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Foul-call direction and prior foul state (clustered logit and game FE LPM)}",
        r"\label{tab:main_regression}",
        r"\begin{threeparttable}",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "c" * len(model_cols) + "}",
        r"\toprule",
        " & ".join([""] + col_headers) + r" \\",
        r"\midrule",
    ]

    for _, row in frame.iterrows():
        label = row["Variable"].replace("−", r"$-$")
        coef_cells = [label]
        se_cells = [""]
        for model in model_cols:
            coef_cells.append(str(row.get(model, "")))
            se_cells.append(str(row.get(f"{model}_se", "")))
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
            r"\item[] All logit specifications include score margin, home possession, period fixed effects, seconds remaining, and season fixed effects.",
            r"\item[] Columns (3)--(4) add bonus indicators; column (4) adds home and away team fixed effects.",
            r"\item[] Column (5) is a linear probability model with game fixed effects and clustered standard errors.",
            r"\item[] Standard errors in parentheses, clustered by game. $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def table2_markdown(path: Path | None = None) -> Path:
    frame = load_table2_frame()
    out = path or Path("docs/tables/table2_main_regression.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    model_cols = [m for m, _ in MODEL_SPECS]
    headers = ["Variable"] + [label for _, label in MODEL_SPECS]
    md_lines = [
        "# Table 2. Main regression results",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        coef = [row["Variable"]] + [str(row.get(m, "")) for m in model_cols]
        se = [""] + [str(row.get(f"{m}_se", "")) for m in model_cols]
        md_lines.append("| " + " | ".join(coef) + " |")
        if any(se[1:]):
            md_lines.append("| " + " | ".join(se) + " |")
    md_lines.extend(
        [
            "",
            "**Notes:** Dependent variable: `foul_against_home`. Standard errors in parentheses, clustered by game.",
            "",
        ]
    )
    out.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return out
