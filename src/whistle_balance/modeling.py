from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

BASELINE_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ score_margin_home_before + home_possession "
    "+ C(period) + seconds_remaining_game + C(season)"
)

SEQUENTIAL_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + time_since_last_foul "
    "+ score_margin_home_before + home_possession "
    "+ C(period) + seconds_remaining_game + C(season)"
)

BONUS_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + time_since_last_foul "
    "+ home_in_bonus_before + away_in_bonus_before "
    "+ score_margin_home_before + home_possession "
    "+ C(period) + seconds_remaining_game + C(season)"
)

TEAM_FE_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + time_since_last_foul "
    "+ home_in_bonus_before + away_in_bonus_before "
    "+ score_margin_home_before + home_possession "
    "+ C(period) + seconds_remaining_game + C(season) "
    "+ C(home_team) + C(away_team)"
)

GAME_FE_FORMULA = (
    "foul_against_home ~ foul_diff_home_minus_away_before "
    "+ period_foul_diff_home_minus_away_before "
    "+ last_foul_against_home + time_since_last_foul "
    "+ home_in_bonus_before + away_in_bonus_before "
    "+ score_margin_home_before + home_possession "
    "+ C(period) + seconds_remaining_game + C(game_id)"
)

BASELINE_REQUIRED_COLUMNS = [
    "foul_against_home",
    "foul_diff_home_minus_away_before",
    "score_margin_home_before",
    "home_possession",
    "period",
    "seconds_remaining_game",
    "season",
    "technical_foul",
    "flagrant_foul",
]

EXTENDED_REQUIRED_COLUMNS = BASELINE_REQUIRED_COLUMNS + [
    "period_foul_diff_home_minus_away_before",
    "last_foul_against_home",
    "time_since_last_foul",
    "home_in_bonus_before",
    "away_in_bonus_before",
    "home_team",
    "away_team",
    "game_id",
]

KEY_COEFFICIENTS = [
    "foul_diff_home_minus_away_before",
    "period_foul_diff_home_minus_away_before",
    "last_foul_against_home",
    "time_since_last_foul",
    "home_in_bonus_before",
    "away_in_bonus_before",
]


@dataclass
class ModelSpec:
    name: str
    formula: str
    required_columns: list[str]


MODEL_SPECS = [
    ModelSpec("baseline", BASELINE_FORMULA, BASELINE_REQUIRED_COLUMNS),
    ModelSpec("sequential", SEQUENTIAL_FORMULA, EXTENDED_REQUIRED_COLUMNS),
    ModelSpec("bonus", BONUS_FORMULA, EXTENDED_REQUIRED_COLUMNS),
    ModelSpec("team_fe", TEAM_FE_FORMULA, EXTENDED_REQUIRED_COLUMNS),
    ModelSpec("game_fe", GAME_FE_FORMULA, EXTENDED_REQUIRED_COLUMNS),
]

ROBUSTNESS_MODEL_SPECS = [
    ModelSpec("baseline", BASELINE_FORMULA, BASELINE_REQUIRED_COLUMNS),
    ModelSpec("sequential", SEQUENTIAL_FORMULA, EXTENDED_REQUIRED_COLUMNS),
    ModelSpec("bonus", BONUS_FORMULA, EXTENDED_REQUIRED_COLUMNS),
]

ROBUSTNESS_TRACKED_TERMS = [
    "foul_diff_home_minus_away_before",
    "period_foul_diff_home_minus_away_before",
    "last_foul_against_home",
]


def _main_foul_filter(data: pd.DataFrame) -> pd.DataFrame:
    """Non-technical, non-flagrant, non-offensive fouls (main JQAS sample)."""
    return data[
        (data["technical_foul"] == 0)
        & (data["flagrant_foul"] == 0)
        & (data["offensive_foul"] == 0)
    ].copy()


def baseline_sample(data: pd.DataFrame) -> pd.DataFrame:
    sample = _main_foul_filter(data)
    return sample.dropna(subset=[c for c in BASELINE_REQUIRED_COLUMNS if c not in {"technical_foul", "flagrant_foul"}])


def extended_sample(data: pd.DataFrame) -> pd.DataFrame:
    sample = _main_foul_filter(data)
    return sample.dropna(
        subset=[c for c in EXTENDED_REQUIRED_COLUMNS if c not in {"technical_foul", "flagrant_foul"}]
    )


def foul_type_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Non-technical, non-flagrant fouls (includes offensive) for type heterogeneity."""
    sample = data[(data["technical_foul"] == 0) & (data["flagrant_foul"] == 0)].copy()
    return sample.dropna(
        subset=[c for c in EXTENDED_REQUIRED_COLUMNS if c not in {"technical_foul", "flagrant_foul"}]
    )


def fit_logit(formula: str, data: pd.DataFrame):
    return smf.logit(formula=formula, data=data).fit(disp=False, maxiter=100)


def fit_baseline_logit(data: pd.DataFrame):
    sample = baseline_sample(data)
    model = fit_logit(BASELINE_FORMULA, sample)
    return model, sample


def extract_key_results(model, model_name: str, n_obs: int) -> list[dict]:
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
                "pseudo_r2": float(getattr(model, "prsquared", float("nan"))),
            }
        )
    return rows


def prepare_model_sample(data: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    sample = _main_foul_filter(data)
    drop_cols = [c for c in spec.required_columns if c not in {"technical_foul", "flagrant_foul"}]
    return sample.dropna(subset=drop_cols)


def run_robustness_checks(
    data: pd.DataFrame,
    specs: dict[str, str],
    model_specs: list[ModelSpec] | None = None,
) -> pd.DataFrame:
    """Fit selected models on each robustness subsample."""
    model_specs = model_specs or ROBUSTNESS_MODEL_SPECS
    rows: list[dict] = []

    for spec_name, query in specs.items():
        for model_spec in model_specs:
            sample = prepare_model_sample(data, model_spec)
            if query:
                subset = sample.query(query)
            else:
                subset = sample

            row = {
                "spec": spec_name,
                "model": model_spec.name,
                "n_obs": len(subset),
                "status": "ok",
            }
            if len(subset) < 100:
                row["status"] = "insufficient sample"
                rows.append(row)
                continue

            try:
                model = fit_logit(model_spec.formula, subset)
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"fit failed: {exc}"
                rows.append(row)
                continue

            row["n_obs"] = int(model.nobs)
            row["pseudo_r2"] = float(getattr(model, "prsquared", float("nan")))
            for term in ROBUSTNESS_TRACKED_TERMS:
                if term in model.params.index:
                    row[f"coef_{term}"] = float(model.params[term])
                    row[f"p_{term}"] = float(model.pvalues[term])
            rows.append(row)

    return pd.DataFrame(rows)


def fit_all_models(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    summaries: list[dict] = []
    models: dict[str, object] = {}

    baseline_model, baseline_data = fit_baseline_logit(data)
    models["baseline"] = baseline_model
    summaries.extend(extract_key_results(baseline_model, "baseline", len(baseline_data)))

    extended_data = extended_sample(data)
    n_games = extended_data["game_id"].nunique()
    for spec in MODEL_SPECS[1:]:
        if spec.name == "game_fe" and n_games > 500:
            continue
        try:
            model = fit_logit(spec.formula, extended_data)
        except (MemoryError, np.linalg.LinAlgError) as exc:
            print(f"Skipping {spec.name}: {exc}")
            continue
        models[spec.name] = model
        summaries.extend(extract_key_results(model, spec.name, len(extended_data)))

    return pd.DataFrame(summaries), models


def summarize_model(model, path=None):
    summary = model.summary().as_text()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary, encoding="utf-8")
    return summary
