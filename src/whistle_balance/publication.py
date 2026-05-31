"""Publication-ready estimation: clustered SE, game FE, margins, placebo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from statsmodels.discrete.conditional_models import ConditionalLogit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from whistle_balance.config import FIGURE_DIR, TABLE_DIR
from whistle_balance.data_utils import ensure_dir
from whistle_balance.modeling import (
    BASELINE_FORMULA,
    BONUS_FORMULA,
    EXTENDED_REQUIRED_COLUMNS,
    KEY_COEFFICIENTS,
    SEQUENTIAL_FORMULA,
    TEAM_FE_FORMULA,
    baseline_sample,
    extended_sample,
    foul_type_sample,
    fit_logit,
    summarize_model,
)

WITHIN_GAME_LPM_VARS = [
    "foul_diff_home_minus_away_before",
    "period_foul_diff_home_minus_away_before",
    "last_foul_against_home",
    "time_since_last_foul",
    "score_margin_home_before",
    "home_possession",
    "seconds_remaining_game",
]

CONDITIONAL_LOGIT_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + score_margin_home_before "
    "+ home_possession + C(period) - 1"
)

INTERACTION_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + time_since_last_foul "
    "+ score_margin_home_before + home_possession "
    "+ C(period) + seconds_remaining_game + C(season) "
    "+ foul_diff_home_minus_away_before:C(playoffs) "
    "+ foul_diff_home_minus_away_before:close_game "
    "+ last_foul_against_home:foul_diff_home_minus_away_before"
)

FOUL_TYPE_FILTERS = {
    "shooting": "shooting_foul == 1",
    "offensive": "offensive_foul == 1",
    "loose_ball": "loose_ball_foul == 1",
    "personal": "foul_type == 'personal'",
}


def fit_logit_clustered(
    formula: str,
    data: pd.DataFrame,
    cluster_col: str = "game_id",
):
    model = smf.logit(formula=formula, data=data)
    return model.fit(
        disp=False,
        maxiter=100,
        cov_type="cluster",
        cov_kwds={"groups": data[cluster_col]},
    )


def extract_logit_rows(model, model_name: str, n_obs: int, n_games: int, se_type: str) -> list[dict]:
    rows: list[dict] = []
    for term in KEY_COEFFICIENTS:
        if term not in model.params.index:
            continue
        rows.append(
            {
                "model": model_name,
                "term": term,
                "coef": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "n_obs": n_obs,
                "n_games": n_games,
                "se_type": se_type,
                "pseudo_r2": float(getattr(model, "prsquared", np.nan)),
            }
        )
    return rows


def run_clustered_main_table(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    specs = {
        "baseline_cluster": (BASELINE_FORMULA, baseline_sample(data)),
        "sequential_cluster": (SEQUENTIAL_FORMULA, extended_sample(data)),
        "bonus_cluster": (BONUS_FORMULA, extended_sample(data)),
        "team_fe_cluster": (TEAM_FE_FORMULA, extended_sample(data)),
    }
    models: dict[str, object] = {}
    for name, (formula, sample) in specs.items():
        model = fit_logit_clustered(formula, sample)
        models[name] = model
        rows.extend(
            extract_logit_rows(model, name, len(sample), sample["game_id"].nunique(), "cluster_game")
        )
        summarize_model(model, TABLE_DIR / f"publication_{name}.txt")
    return pd.DataFrame(rows), models


def run_game_fe_lpm(data: pd.DataFrame) -> pd.DataFrame:
    sample = extended_sample(data).sort_values(["game_id", "period", "seconds_remaining_game"])
    sample["event_idx"] = sample.groupby("game_id").cumcount()
    panel = sample.set_index(["game_id", "event_idx"])

    y = panel["foul_against_home"]
    x_cols = WITHIN_GAME_LPM_VARS + ["period"]
    X = pd.get_dummies(panel[x_cols], columns=["period"], drop_first=True)
    result = PanelOLS(y, X, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)

    rows = []
    for term in KEY_COEFFICIENTS:
        if term not in result.params.index:
            continue
        rows.append(
            {
                "model": "game_fe_lpm",
                "term": term,
                "coef": float(result.params[term]),
                "std_err": float(result.std_errors[term]),
                "p_value": float(result.pvalues[term]),
                "n_obs": int(result.nobs),
                "n_games": sample["game_id"].nunique(),
                "se_type": "cluster_game",
                "pseudo_r2": float(result.rsquared_within),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "publication_game_fe_lpm.csv", index=False)
    Path(TABLE_DIR / "publication_game_fe_lpm.txt").write_text(str(result.summary), encoding="utf-8")
    return pd.DataFrame(rows)


def run_conditional_logit_subsample(
    data: pd.DataFrame,
    n_games: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    sample = extended_sample(data)
    chosen = sample["game_id"].drop_duplicates().sample(n=n_games, random_state=random_state)
    sub = sample[sample["game_id"].isin(chosen)].copy()
    try:
        y, X = patsy.dmatrices(CONDITIONAL_LOGIT_FORMULA, sub, return_type="dataframe")
        model = ConditionalLogit(y.iloc[:, 0], X, groups=sub["game_id"].values)
        result = model.fit(disp=False, maxiter=80)
    except (ValueError, np.linalg.LinAlgError) as exc:
        out = pd.DataFrame(
            [{"model": "conditional_logit_subsample", "term": "status", "coef": np.nan, "note": str(exc)}]
        )
        out.to_csv(TABLE_DIR / "publication_conditional_logit_subsample.csv", index=False)
        print(f"Conditional logit subsample skipped: {exc}")
        return out

    rows = []
    for term in KEY_COEFFICIENTS:
        if term not in result.params.index:
            continue
        rows.append(
            {
                "model": "conditional_logit_subsample",
                "term": term,
                "coef": float(result.params[term]),
                "std_err": float(result.bse[term]),
                "p_value": float(result.pvalues[term]),
                "n_obs": len(sub),
                "n_games": sub["game_id"].nunique(),
                "se_type": "conditional_mle",
                "pseudo_r2": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "publication_conditional_logit_subsample.csv", index=False)
    summarize_model(result, TABLE_DIR / "publication_conditional_logit_subsample.txt")
    return out


def _representative_row(sample: pd.DataFrame) -> pd.DataFrame:
    """One template row at typical game-state values for counterfactual prediction."""
    mask = (
        (sample["period"] == sample["period"].mode().iloc[0])
        & (sample["season"] == sample["season"].mode().iloc[0])
        & sample["last_foul_against_home"].notna()
    )
    template = sample.loc[mask].head(1).copy()
    if template.empty:
        template = sample.head(1).copy()
    numeric_medians = sample.median(numeric_only=True)
    for col in numeric_medians.index:
        if col in template.columns:
            template[col] = numeric_medians[col]
    template["period"] = int(sample["period"].mode().iloc[0])
    template["season"] = int(sample["season"].mode().iloc[0])
    if "home_in_bonus_before" in template.columns:
        template["home_in_bonus_before"] = 0
    if "away_in_bonus_before" in template.columns:
        template["away_in_bonus_before"] = 0
    return template


def _predict_one(model, frame: pd.DataFrame) -> float:
    return float(np.asarray(model.predict(frame)).ravel()[0])


def compute_logit_ame(model, data: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Average marginal effects for logit: E[p(1-p)] * beta."""
    probs = np.asarray(model.predict(data))
    scale = float(np.mean(probs * (1.0 - probs)))
    rows: list[dict] = []
    for term in terms:
        if term not in model.params.index:
            continue
        rows.append(
            {
                "term": term,
                "dy_dx": float(model.params[term] * scale),
                "std_err": float(abs(scale) * model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ame_scale": scale,
            }
        )
    return pd.DataFrame(rows)


def predicted_probability_table(model, sample: pd.DataFrame, max_rows: int = 20000) -> pd.DataFrame:
    """Average model predictions over a subsample at each counterfactual grid point."""
    if len(sample) > max_rows:
        base = sample.sample(max_rows, random_state=42).copy()
    else:
        base = sample.copy()

    rows: list[dict] = []
    for foul_diff in range(-3, 4):
        for last_foul in (0, 1):
            frame = base.copy()
            frame["foul_diff_home_minus_away_before"] = foul_diff
            frame["period_foul_diff_home_minus_away_before"] = foul_diff
            frame["last_foul_against_home"] = last_foul
            prob = float(np.asarray(model.predict(frame)).mean())
            rows.append(
                {
                    "foul_diff_home_minus_away_before": foul_diff,
                    "last_foul_against_home": last_foul,
                    "predicted_p_foul_against_home": prob,
                }
            )
    return pd.DataFrame(rows)


def discrete_change_table(model, sample: pd.DataFrame, max_rows: int = 20000) -> pd.DataFrame:
    """Average discrete change when foul_diff increases by 1."""
    if len(sample) > max_rows:
        base = sample.sample(max_rows, random_state=42).copy()
    else:
        base = sample.copy()

    rows: list[dict] = []
    for last_foul in (0, 1):
        frame0 = base.copy()
        frame0["last_foul_against_home"] = last_foul
        frame0["foul_diff_home_minus_away_before"] = 0
        frame0["period_foul_diff_home_minus_away_before"] = 0
        p0 = float(np.asarray(model.predict(frame0)).mean())

        frame1 = base.copy()
        frame1["last_foul_against_home"] = last_foul
        frame1["foul_diff_home_minus_away_before"] = 1
        frame1["period_foul_diff_home_minus_away_before"] = 1
        p1 = float(np.asarray(model.predict(frame1)).mean())

        rows.append(
            {
                "last_foul_against_home": last_foul,
                "delta_foul_diff": 1,
                "p_at_diff_0": p0,
                "p_at_diff_1": p1,
                "discrete_change": p1 - p0,
            }
        )
    return pd.DataFrame(rows)


def run_marginal_effects(models: dict[str, object], sample: pd.DataFrame) -> pd.DataFrame:
    ext = extended_sample(sample)
    rows: list[dict] = []
    for name, model in models.items():
        if "cluster" not in name:
            continue
        ame = compute_logit_ame(model, ext, KEY_COEFFICIENTS)
        ame.insert(0, "model", name)
        rows.append(ame)

    margin_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    margin_df.to_csv(TABLE_DIR / "publication_marginal_effects.csv", index=False)

    pred_model = models.get("sequential_cluster")
    if pred_model is not None:
        pred = predicted_probability_table(pred_model, ext)
        pred.to_csv(TABLE_DIR / "publication_predicted_probabilities.csv", index=False)
        discrete_change_table(pred_model, ext).to_csv(
            TABLE_DIR / "publication_discrete_changes.csv", index=False
        )
        _plot_predicted_probabilities(pred)
    return margin_df


def _plot_predicted_probabilities(pred: pd.DataFrame) -> None:
    from whistle_balance.plot_style import COLORS, add_half_line, apply_plot_style, save_figure, style_axes

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for last_foul, label, color in [
        (0, "Previous foul on away", COLORS["away"]),
        (1, "Previous foul on home", COLORS["home"]),
    ]:
        sub = pred[pred["last_foul_against_home"] == last_foul]
        ax.plot(
            sub["foul_diff_home_minus_away_before"],
            sub["predicted_p_foul_against_home"],
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            color=color,
            linewidth=2.0,
        )
    add_half_line(ax)
    ax.set_xlabel("Home minus away fouls before event")
    ax.set_ylabel("Predicted P(next foul against home)")
    ax.set_title("Model-predicted call direction by foul differential", pad=10)
    ax.legend(loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax)
    save_figure(fig, "publication_predicted_probabilities.png")


def _recompute_sequential_state(game: pd.DataFrame) -> pd.DataFrame:
    game = game.sort_values("_shuffle_order").copy()
    home_fouls = away_fouls = 0
    home_period = away_period = 0
    current_period = None
    last_foul_against_home = np.nan

    home_fouls_list = []
    away_fouls_list = []
    foul_diff_list = []
    home_period_list = []
    away_period_list = []
    period_diff_list = []
    last_foul_list = []

    for row in game.itertuples(index=False):
        period = int(row.period)
        if current_period != period:
            home_period = away_period = 0
            current_period = period

        home_fouls_list.append(home_fouls)
        away_fouls_list.append(away_fouls)
        foul_diff_list.append(home_fouls - away_fouls)
        home_period_list.append(home_period)
        away_period_list.append(away_period)
        period_diff_list.append(home_period - away_period)
        last_foul_list.append(last_foul_against_home)

        if row.technical_foul == 0 and row.flagrant_foul == 0:
            if row.foul_against_home == 1:
                home_fouls += 1
                home_period += 1
            else:
                away_fouls += 1
                away_period += 1
            last_foul_against_home = row.foul_against_home

    game["foul_diff_home_minus_away_before"] = foul_diff_list
    game["period_foul_diff_home_minus_away_before"] = period_diff_list
    game["last_foul_against_home"] = last_foul_list
    return game


def build_placebo_sample(data: pd.DataFrame, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    sample = extended_sample(data).copy()
    parts: list[pd.DataFrame] = []
    for _, game in sample.groupby("game_id", sort=False):
        game = game.copy()
        game["_shuffle_order"] = rng.permutation(len(game))
        parts.append(_recompute_sequential_state(game))
    return pd.concat(parts, ignore_index=True)


def run_placebo_tests(data: pd.DataFrame, n_draws: int = 100, random_state: int = 42) -> pd.DataFrame:
    ext = extended_sample(data)
    actual = fit_logit_clustered(SEQUENTIAL_FORMULA, ext)
    rows: list[dict] = [
        {
            "draw": "actual",
            "coef_foul_diff": float(actual.params["foul_diff_home_minus_away_before"]),
            "coef_period_diff": float(actual.params["period_foul_diff_home_minus_away_before"]),
            "coef_last_foul": float(actual.params["last_foul_against_home"]),
        }
    ]

    for draw in range(n_draws):
        print(f"Placebo draw {draw + 1}/{n_draws}...", flush=True)
        placebo = build_placebo_sample(data, random_state=random_state + draw)
        model = fit_logit(SEQUENTIAL_FORMULA, placebo)
        rows.append(
            {
                "draw": draw,
                "coef_foul_diff": float(model.params["foul_diff_home_minus_away_before"]),
                "coef_period_diff": float(model.params["period_foul_diff_home_minus_away_before"]),
                "coef_last_foul": float(model.params["last_foul_against_home"]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "publication_placebo_draws.csv", index=False)
    placebo_only = out[out["draw"] != "actual"]
    summary = placebo_only[["coef_foul_diff", "coef_period_diff", "coef_last_foul"]].agg(["mean", "std"])
    summary.to_csv(TABLE_DIR / "publication_placebo_summary.csv")
    print("Placebo means:", summary.loc["mean"].to_dict())
    return out


def run_interaction_models(data: pd.DataFrame) -> pd.DataFrame:
    sample = extended_sample(data).copy()
    sample["close_game"] = (sample["score_margin_home_before"].abs() <= 10).astype(int)
    model = fit_logit_clustered(INTERACTION_FORMULA, sample)
    summarize_model(model, TABLE_DIR / "publication_interactions.txt")

    tracked = [
        "foul_diff_home_minus_away_before",
        "period_foul_diff_home_minus_away_before",
        "last_foul_against_home",
        "foul_diff_home_minus_away_before:C(playoffs)[T.1]",
        "foul_diff_home_minus_away_before:close_game",
        "last_foul_against_home:foul_diff_home_minus_away_before",
    ]
    rows = []
    for term in tracked:
        if term not in model.params.index:
            continue
        rows.append(
            {
                "term": term,
                "coef": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "publication_interactions.csv", index=False)
    return out


def run_foul_type_models(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    base = foul_type_sample(data)
    for label, query in FOUL_TYPE_FILTERS.items():
        subset = base.query(query)
        if len(subset) < 500:
            rows.append({"foul_type": label, "n_obs": len(subset), "status": "insufficient"})
            continue
        model = fit_logit_clustered(SEQUENTIAL_FORMULA, subset)
        rows.append(
            {
                "foul_type": label,
                "n_obs": int(model.nobs),
                "status": "ok",
                "coef_foul_diff": float(model.params["foul_diff_home_minus_away_before"]),
                "p_foul_diff": float(model.pvalues["foul_diff_home_minus_away_before"]),
                "coef_period_diff": float(model.params["period_foul_diff_home_minus_away_before"]),
                "p_period_diff": float(model.pvalues["period_foul_diff_home_minus_away_before"]),
                "coef_last_foul": float(model.params["last_foul_against_home"]),
                "se_last_foul": float(model.bse["last_foul_against_home"]),
                "p_last_foul": float(model.pvalues["last_foul_against_home"]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_DIR / "publication_foul_type_models.csv", index=False)
    return out


def run_publication_analysis(data: pd.DataFrame, *, skip_conditional_logit: bool = True) -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)

    clustered, models = run_clustered_main_table(data)
    clustered.to_csv(TABLE_DIR / "publication_clustered_main.csv", index=False)

    game_fe = run_game_fe_lpm(data)
    cond = (
        pd.DataFrame()
        if skip_conditional_logit
        else run_conditional_logit_subsample(data)
    )
    run_marginal_effects(models, data)
    run_interaction_models(data)
    run_foul_type_models(data)
    placebo = run_placebo_tests(data)

    master_parts = [clustered, game_fe]
    if not cond.empty:
        master_parts.append(cond)
    master = pd.concat(master_parts, ignore_index=True)
    if "term" in master.columns:
        master = master[master["term"] != "status"]
    master.to_csv(TABLE_DIR / "publication_master_coefficients.csv", index=False)

    print("Publication analysis complete.")
    print(f"  publication_clustered_main.csv ({len(clustered)} rows)")
    print(f"  publication_game_fe_lpm.csv")
    print(f"  publication_marginal_effects.csv")
    print(f"  publication_predicted_probabilities.csv")
    print(f"  publication_discrete_changes.csv")
    print(f"  publication_placebo_draws.csv ({len(placebo)} draws)")
    print(f"  publication_master_coefficients.csv ({len(master)} rows)")
