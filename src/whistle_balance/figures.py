"""Publication figures beyond basic descriptive plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import blended_transform_factory

from matplotlib.gridspec import GridSpec

from whistle_balance.config import FIGURE_DIR, TABLE_DIR
from whistle_balance.data_utils import ensure_dir
from whistle_balance.descriptive import (
    extended_descriptive_sample,
    foul_diff_next_call_table,
    period_foul_diff_next_call_table,
)
from whistle_balance.modeling import SEQUENTIAL_FORMULA, baseline_sample, extended_sample
from whistle_balance.plot_style import (
    COLORS,
    TERM_COLORS,
    add_half_line,
    add_panel_label,
    add_zero_line,
    annotate_bars,
    apply_plot_style,
    proportion_ci_table,
    save_figure,
    style_axes,
    wilson_interval,
)
from whistle_balance.publication import fit_logit_clustered, predicted_probability_table

apply_plot_style()

KEY_TERMS = [
    "foul_diff_home_minus_away_before",
    "period_foul_diff_home_minus_away_before",
    "last_foul_against_home",
]

FIGURE4_TERMS = [
    "last_foul_against_home",
    "period_foul_diff_home_minus_away_before",
    "foul_diff_home_minus_away_before",
]

FIGURE4_FORMAL_LABELS = {
    "foul_diff_home_minus_away_before": "Game foul\ndifferential",
    "period_foul_diff_home_minus_away_before": "Period foul\ndifferential",
    "last_foul_against_home": "Previous foul\non home",
}

FIGURE4_SHORT_LABELS = {
    "foul_diff_home_minus_away_before": "Game diff",
    "period_foul_diff_home_minus_away_before": "Period diff",
    "last_foul_against_home": "Previous call",
}

TERM_LABELS = {
    "foul_diff_home_minus_away_before": "Game foul diff",
    "period_foul_diff_home_minus_away_before": "Period foul diff",
    "last_foul_against_home": "Last foul on home",
}

MODEL_LABELS = {
    "baseline_cluster": "Baseline",
    "sequential_cluster": "Sequential",
    "bonus_cluster": "Bonus controls",
    "team_fe_cluster": "Team FE",
    "game_fe_lpm": "Game FE (LPM)",
}

FOUL_TYPE_PANELS = [
    ("shooting_foul == 1", "Shooting fouls"),
    ("foul_type == 'personal'", "Personal fouls"),
    ("offensive_foul == 1", "Offensive fouls"),
    ("loose_ball_foul == 1", "Loose-ball fouls"),
]


def _save(fig: plt.Figure, name: str) -> None:
    save_figure(fig, name)


def _last_foul_table(sample: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sample.groupby("last_foul_against_home", observed=True)["foul_against_home"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "p", "count": "n"})
    )
    grouped["last_foul_against_home"] = grouped["last_foul_against_home"].astype(int)
    intervals = [
        wilson_interval(row.p * row.n, row.n)
        for row in grouped.itertuples(index=False)
    ]
    grouped["ci_low"] = [lo for lo, _ in intervals]
    grouped["ci_high"] = [hi for _, hi in intervals]
    return grouped


def _plot_last_foul_bars(ax: plt.Axes, sample: pd.DataFrame, *, title: str | None = None) -> None:
    table = _last_foul_table(sample)
    labels = ["Prev. foul\non away", "Prev. foul\non home"]
    colors = [COLORS["away"], COLORS["home"]]
    x = np.arange(2)
    bars = ax.bar(
        x,
        table.sort_values("last_foul_against_home")["p"],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.62,
        zorder=3,
    )
    sorted_table = table.sort_values("last_foul_against_home")
    ax.errorbar(
        x,
        sorted_table["p"],
        yerr=[
            sorted_table["p"] - sorted_table["ci_low"],
            sorted_table["ci_high"] - sorted_table["p"],
        ],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=4,
        capthick=1.0,
        linewidth=1.0,
        zorder=4,
    )
    add_half_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(next foul against home)")
    if title:
        ax.set_title(title, pad=8)
    style_axes(ax)
    annotate_bars(ax, bars, sorted_table["p"].values, fmt="{:.1%}", offset=0.008)


def _plot_binned_probability_line(
    ax: plt.Axes,
    sample: pd.DataFrame,
    bin_col: str,
    *,
    clip: tuple[int, int],
    xlabel: str,
    title: str | None = None,
    color: str = COLORS["secondary"],
) -> None:
    sub = sample.copy()
    sub["bin"] = sub[bin_col].clip(*clip)
    grouped = (
        sub.groupby("bin", observed=True)["foul_against_home"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = proportion_ci_table(grouped)
    ax.fill_between(
        grouped["bin"],
        grouped["ci_low"],
        grouped["ci_high"],
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        grouped["bin"],
        grouped["mean"],
        marker="o",
        markersize=5.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        color=color,
        linewidth=2.0,
        zorder=3,
    )
    add_half_line(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(next foul against home)")
    if title:
        ax.set_title(title, pad=8)
    style_axes(ax)


def _plot_forest_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    *,
    title: str,
    color: str,
    xlabel: str = "Estimate (95% CI)",
) -> None:
    y_pos = np.arange(len(df))
    labels = [MODEL_LABELS.get(m, m) for m in df["model"]]
    ax.errorbar(
        df["coef"],
        y_pos,
        xerr=1.96 * df["std_err"],
        fmt="o",
        color=color,
        ecolor=color,
        capsize=4,
        capthick=1.0,
        markersize=6.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linewidth=1.2,
        zorder=3,
    )
    add_zero_line(ax)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    style_axes(ax, ygrid=False)


def plot_coefficient_forest() -> None:
    clustered = pd.read_csv(TABLE_DIR / "publication_clustered_main.csv")
    game_fe = pd.read_csv(TABLE_DIR / "publication_game_fe_lpm.csv")
    game_fe["model"] = "game_fe_lpm"
    df = pd.concat([clustered, game_fe], ignore_index=True)
    df = df[df["term"].isin(KEY_TERMS)]

    model_order = [
        "baseline_cluster",
        "sequential_cluster",
        "bonus_cluster",
        "team_fe_cluster",
        "game_fe_lpm",
    ]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["term", "model"])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    for ax, term in zip(axes, KEY_TERMS, strict=True):
        sub = df[df["term"] == term]
        _plot_forest_panel(ax, sub, title=TERM_LABELS[term], color=TERM_COLORS[term])
    fig.suptitle(
        "Main coefficients across model specifications",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "publication_coefficient_forest.png")


def _actual_last_foul_coef() -> tuple[float, float]:
    """Current sequential clustered estimate for last_foul_against_home."""
    main = pd.read_csv(TABLE_DIR / "publication_clustered_main.csv")
    row = main[
        (main["model"] == "sequential_cluster")
        & (main["term"] == "last_foul_against_home")
    ].iloc[0]
    return float(row["coef"]), float(row["std_err"])


def plot_figure_placebo() -> None:
    """Main-text falsification figure: actual vs within-game shuffle placebo."""
    draws = pd.read_csv(TABLE_DIR / "publication_placebo_draws.csv")
    placebo = draws.loc[draws["draw"] != "actual", "coef_last_foul"].astype(float)
    actual, _ = _actual_last_foul_coef()
    placebo_mean = float(placebo.mean())
    emp_p = int((placebo <= actual).sum())

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.hist(
        placebo,
        bins=min(20, max(8, len(placebo) // 5)),
        density=True,
        color=COLORS["placebo"],
        edgecolor="white",
        linewidth=1.0,
        alpha=0.92,
        zorder=2,
        label=f"Placebo draws ($n={len(placebo)}$)",
    )
    ax.axvline(
        actual,
        color=COLORS["home"],
        linewidth=2.6,
        label=f"Actual estimate ({actual:.3f})",
        zorder=4,
    )
    ax.axvline(
        placebo_mean,
        color=COLORS["primary"],
        linewidth=1.8,
        linestyle="--",
        label=f"Placebo mean ({placebo_mean:.3f})",
        zorder=3,
    )
    add_zero_line(ax)
    ax.set_xlabel("Coefficient on last foul against home")
    ax.set_ylabel("Density")
    ax.set_title(
        "Destroying within-game foul order eliminates previous-call reversal",
        pad=10,
    )
    ax.text(
        0.02,
        0.97,
        f"Empirical $p = {emp_p}/{len(placebo)}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=COLORS["primary"],
    )
    ax.legend(loc="upper right", frameon=True, edgecolor=COLORS["grid"], fontsize=8)
    style_axes(ax, ygrid=False)
    _save(fig, "publication_figure_placebo.png")


def plot_figure_foul_type_coef() -> None:
    """Main-text foul-type coefficient plot highlighting offensive-foul exception."""
    path = TABLE_DIR / "publication_foul_type_models.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    order = ["shooting", "loose_ball", "personal", "offensive"]
    labels = {
        "shooting": "Shooting fouls",
        "loose_ball": "Loose-ball fouls",
        "personal": "Personal fouls",
        "offensive": "Offensive fouls",
    }
    df["foul_type"] = pd.Categorical(df["foul_type"], categories=order, ordered=True)
    df = df.sort_values("foul_type")
    y = np.arange(len(df))
    colors = [
        COLORS["secondary"] if ft != "offensive" else COLORS["playoffs"]
        for ft in df["foul_type"]
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    if "se_last_foul" in df.columns:
        xerr = 1.96 * df["se_last_foul"].fillna(0)
    else:
        xerr = None
    ax.errorbar(
        df["coef_last_foul"],
        y,
        xerr=xerr,
        fmt="o",
        markersize=8,
        color=COLORS["primary"],
        ecolor=COLORS["neutral"],
        capsize=4,
        linewidth=1.2,
        zorder=3,
    )
    for yi, coef, color in zip(y, df["coef_last_foul"], colors, strict=True):
        ax.scatter(coef, yi, s=120, color=color, edgecolor="white", linewidth=1.2, zorder=4)
    add_zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[ft] for ft in df["foul_type"]])
    ax.set_xlabel("Logit coefficient on last foul against home (95% CI)")
    ax.set_title(
        "Previous-call reversal by foul type (offensive fouls excluded from main sample)",
        pad=10,
    )
    for yi, coef in zip(y, df["coef_last_foul"], strict=True):
        ax.annotate(
            f"{coef:.2f}",
            (coef, yi),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontsize=8,
            color=COLORS["primary"],
        )
    style_axes(ax, ygrid=True)
    _save(fig, "publication_figure_foul_type_coef.png")


def plot_placebo_last_foul() -> None:
    plot_figure_placebo()


def plot_foul_type_heatmap() -> None:
    df = pd.read_csv(TABLE_DIR / "publication_foul_type_models.csv")
    df = df[df["status"] == "ok"].set_index("foul_type")
    matrix = df[["coef_foul_diff", "coef_period_diff", "coef_last_foul"]].astype(float)
    matrix.index = matrix.index.str.replace("_", " ").str.title()
    matrix.columns = ["Game foul diff", "Period foul diff", "Last foul on home"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    vmax = max(abs(matrix.values.min()), abs(matrix.values.max()), 0.05)
    im = ax.imshow(matrix.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=12, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            text_color = "white" if abs(val) > vmax * 0.55 else COLORS["primary"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Logit coefficient", rotation=90, va="center")
    cbar.outline.set_linewidth(0.6)
    ax.set_title("Sequential model coefficients by foul type", pad=10)
    style_axes(ax, ygrid=False)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    _save(fig, "publication_foul_type_heatmap.png")


def plot_playoffs_vs_regular(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for ax, playoffs, title in zip(axes, [0, 1], ["Regular season", "Playoffs"], strict=True):
        _plot_last_foul_bars(ax, sample[sample["playoffs"] == playoffs], title=title)
        ax.set_ylim(0.34, 0.66)

    fig.suptitle(
        "Previous call direction vs next call, by season type",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    _save(fig, "publication_playoffs_vs_regular_last_foul.png")

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for playoffs, label, color in [
        (0, "Regular season", COLORS["regular"]),
        (1, "Playoffs", COLORS["playoffs"]),
    ]:
        sub = sample[sample["playoffs"] == playoffs].copy()
        sub["bin"] = sub["period_foul_diff_home_minus_away_before"].clip(-4, 4)
        grouped = (
            sub.groupby("bin", observed=True)["foul_against_home"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = proportion_ci_table(grouped)
        ax.fill_between(
            grouped["bin"],
            grouped["ci_low"],
            grouped["ci_high"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            grouped["bin"],
            grouped["mean"],
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            color=color,
            linewidth=2.0,
        )
    add_half_line(ax)
    ax.set_xlabel("Home minus away period fouls before event")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Period foul differential vs next call direction", pad=10)
    ax.legend(loc="upper right", frameon=True, fancybox=False, edgecolor=COLORS["grid"])
    style_axes(ax)
    _save(fig, "publication_playoffs_vs_regular_period_diff.png")


def plot_season_coefficients(df: pd.DataFrame) -> None:
    sample = extended_sample(df)
    rows: list[dict] = []
    for season in sorted(sample["season"].unique()):
        sub = sample[sample["season"] == season]
        if len(sub) < 5000:
            continue
        model = fit_logit_clustered(SEQUENTIAL_FORMULA, sub)
        for term in KEY_TERMS:
            if term not in model.params.index:
                continue
            rows.append(
                {
                    "season": int(season),
                    "term": term,
                    "coef": float(model.params[term]),
                    "std_err": float(model.bse[term]),
                }
            )

    coef_df = pd.DataFrame(rows)
    coef_df.to_csv(TABLE_DIR / "publication_season_coefficients.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), sharex=True)
    for ax, term in zip(axes, KEY_TERMS, strict=True):
        sub = coef_df[coef_df["term"] == term]
        color = TERM_COLORS[term]
        ax.errorbar(
            sub["season"],
            sub["coef"],
            yerr=1.96 * sub["std_err"],
            fmt="o-",
            capsize=4,
            color=color,
            ecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.8,
            markersize=6,
        )
        add_zero_line(ax)
        ax.set_title(TERM_LABELS[term], pad=8)
        ax.set_xlabel("Season (end year)")
        ax.set_ylabel("Coefficient")
        style_axes(ax)
    fig.suptitle(
        "Sequential model coefficients by season",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.03,
    )
    _save(fig, "publication_season_coefficients.png")


def plot_descriptive_vs_model(df: pd.DataFrame, sequential_model) -> None:
    desc = foul_diff_next_call_table(df)
    desc = proportion_ci_table(desc.rename(columns={"p_foul_against_home": "mean", "n_fouls": "count"}))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    ax.bar(
        desc["foul_diff_bin"],
        desc["mean"],
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=1.0,
        width=0.82,
        zorder=2,
    )
    ax.errorbar(
        desc["foul_diff_bin"],
        desc["mean"],
        yerr=[desc["mean"] - desc["ci_low"], desc["ci_high"] - desc["mean"]],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=3,
        linewidth=0.9,
        zorder=3,
    )
    add_half_line(ax)
    ax.set_xlabel("Game foul diff bin")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Raw binned rates", pad=8)
    ax.set_ylim(0.38, 0.62)
    style_axes(ax)

    ax = axes[1]
    if sequential_model is not None:
        model_pred = predicted_probability_table(
            sequential_model, extended_sample(df), max_rows=15000
        )
        model_pred = model_pred[model_pred["last_foul_against_home"] == 0]
        ax.plot(
            model_pred["foul_diff_home_minus_away_before"],
            model_pred["predicted_p_foul_against_home"],
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            color=COLORS["home"],
            linewidth=2.0,
        )
    add_half_line(ax)
    ax.set_xlabel("Game foul diff")
    ax.set_ylabel("Model-predicted P(next foul against home)")
    ax.set_title("Adjusted predictions (previous foul on away)", pad=8)
    ax.set_ylim(0.38, 0.62)
    style_axes(ax)

    fig.suptitle(
        "Descriptive vs model-implied foul-differential gradient",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    _save(fig, "publication_descriptive_vs_model.png")


def plot_period_heterogeneity(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    rows = []
    for period in sorted(sample["period"].unique()):
        sub = sample[sample["period"] == period]
        for last in (0, 1):
            p = sub.loc[sub["last_foul_against_home"] == last, "foul_against_home"].mean()
            rows.append({"period": period, "last_foul_against_home": last, "p": p})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    width = 0.36
    periods = sorted(plot_df["period"].unique())
    x = np.arange(len(periods))
    away = plot_df[plot_df["last_foul_against_home"] == 0].set_index("period").loc[periods, "p"]
    home = plot_df[plot_df["last_foul_against_home"] == 1].set_index("period").loc[periods, "p"]
    ax.bar(
        x - width / 2,
        away.values,
        width,
        label="Prev. foul on away",
        color=COLORS["away"],
        edgecolor="white",
        linewidth=1.0,
    )
    ax.bar(
        x + width / 2,
        home.values,
        width,
        label="Prev. foul on home",
        color=COLORS["home"],
        edgecolor="white",
        linewidth=1.0,
    )
    add_half_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{int(p)}" if p <= 4 else f"OT{int(p) - 4}" for p in periods])
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Previous call direction by period", pad=10)
    ax.legend(loc="upper right")
    style_axes(ax)
    _save(fig, "publication_period_heterogeneity.png")


def plot_score_margin_heterogeneity(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample["margin_bin"] = pd.cut(
        sample["score_margin_home_before"].abs(),
        bins=[0, 5, 10, 20, 100],
        labels=["0–5", "6–10", "11–20", "21+"],
    )
    palette = [COLORS["secondary"], COLORS["away"], COLORS["accent"], COLORS["home"]]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for margin_bin, color in zip(["0–5", "6–10", "11–20", "21+"], palette, strict=True):
        sub = sample[sample["margin_bin"] == margin_bin].copy()
        sub["bin"] = sub["period_foul_diff_home_minus_away_before"].clip(-3, 3)
        grouped = (
            sub.groupby("bin", observed=True)["foul_against_home"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = proportion_ci_table(grouped)
        ax.plot(
            grouped["bin"],
            grouped["mean"],
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=f"|margin| {margin_bin}",
            color=color,
            linewidth=1.8,
        )
    add_half_line(ax)
    ax.set_xlabel("Period foul diff bin")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Period foul diff vs next call, by score margin", pad=10)
    ax.legend(title="Score margin", loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax)
    _save(fig, "publication_score_margin_heterogeneity.png")


def plot_bonus_threshold(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    specs = [
        ("away_period_fouls_before", "Away period fouls (home near bonus)", COLORS["home"]),
        ("home_period_fouls_before", "Home period fouls (away near bonus)", COLORS["away"]),
    ]
    for col, label, color in specs:
        sub = sample[sample[col].between(2, 7)].copy()
        grouped = (
            sub.groupby(col, observed=True)["foul_against_home"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = proportion_ci_table(grouped)
        ax.fill_between(
            grouped[col],
            grouped["ci_low"],
            grouped["ci_high"],
            color=color,
            alpha=0.15,
            linewidth=0,
        )
        ax.plot(
            grouped[col],
            grouped["mean"],
            marker="o",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            color=color,
            linewidth=2.0,
        )
    ax.axvline(4.5, color=COLORS["zero"], linestyle=":", linewidth=1.2, label="Bonus threshold (~5)")
    add_half_line(ax)
    ax.set_xlabel("Opponent period fouls before event")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Call direction near bonus threshold", pad=10)
    ax.legend(fontsize=8, loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax)
    _save(fig, "publication_bonus_threshold.png")


def plot_time_since_last_foul(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample = sample[sample["time_since_last_foul"].notna()]
    sample["time_bin"] = pd.cut(
        sample["time_since_last_foul"],
        bins=[0, 15, 30, 60, 120, 300, 720],
        labels=["0–15s", "16–30s", "31–60s", "1–2m", "2–5m", "5m+"],
    )
    grouped = (
        sample.groupby("time_bin", observed=True)["foul_against_home"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = proportion_ci_table(grouped)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(grouped))
    bars = ax.bar(
        x,
        grouped["mean"],
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=1.0,
        width=0.72,
        zorder=2,
    )
    ax.errorbar(
        x,
        grouped["mean"],
        yerr=[grouped["mean"] - grouped["ci_low"], grouped["ci_high"] - grouped["mean"]],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=4,
        linewidth=1.0,
        zorder=3,
    )
    add_half_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["time_bin"], rotation=18, ha="right")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Next call direction by time since previous foul", pad=10)
    style_axes(ax)
    annotate_bars(ax, bars, grouped["mean"].values, fmt="{:.1%}", offset=0.006)
    _save(fig, "publication_time_since_last_foul.png")


def plot_marginal_effects() -> None:
    df = pd.read_csv(TABLE_DIR / "publication_marginal_effects.csv")
    df = df[df["term"].isin(KEY_TERMS)]
    model_order = ["baseline_cluster", "sequential_cluster", "bonus_cluster", "team_fe_cluster"]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["term", "model"])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), sharey=True)
    for ax, term in zip(axes, KEY_TERMS, strict=True):
        sub = df[df["term"] == term]
        y_pos = np.arange(len(sub))
        labels = [MODEL_LABELS.get(m, m) for m in sub["model"]]
        color = TERM_COLORS[term]
        ax.errorbar(
            sub["dy_dx"],
            y_pos,
            xerr=1.96 * sub["std_err"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.2,
        )
        add_zero_line(ax)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_title(TERM_LABELS[term], pad=8)
        ax.set_xlabel("Average marginal effect (95% CI)")
        style_axes(ax, ygrid=False)
    fig.suptitle(
        "Average marginal effects on P(next foul against home)",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    _save(fig, "publication_marginal_effects.png")


def plot_placebo_all_terms() -> None:
    draws = pd.read_csv(TABLE_DIR / "publication_placebo_draws.csv")
    actual = draws[draws["draw"] == "actual"].iloc[0]
    placebo = draws[draws["draw"] != "actual"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    specs = [
        ("coef_foul_diff", "Game foul diff", float(actual["coef_foul_diff"]), TERM_COLORS["foul_diff_home_minus_away_before"]),
        ("coef_period_diff", "Period foul diff", float(actual["coef_period_diff"]), TERM_COLORS["period_foul_diff_home_minus_away_before"]),
        ("coef_last_foul", "Last foul on home", float(actual["coef_last_foul"]), TERM_COLORS["last_foul_against_home"]),
    ]
    for ax, (col, title, actual_val, color) in zip(axes, specs, strict=True):
        ax.hist(
            placebo[col],
            bins=8,
            color=COLORS["placebo"],
            edgecolor="white",
            linewidth=1.0,
            zorder=2,
        )
        ax.axvline(actual_val, color=color, linewidth=2.2, label=f"Actual ({actual_val:.3f})", zorder=4)
        add_zero_line(ax)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Count")
        ax.set_title(title, pad=8)
        ax.legend(fontsize=8, loc="upper right")
        style_axes(ax, ygrid=False)
    fig.suptitle(
        "Placebo tests: within-game shuffled foul order",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.03,
    )
    _save(fig, "publication_placebo_all_terms.png")


def plot_interaction_coefficients() -> None:
    df = pd.read_csv(TABLE_DIR / "publication_interactions.csv")
    main = df[~df["term"].str.contains(":", regex=False)].copy()
    interact = df[df["term"].str.contains(":", regex=False)].copy()
    interact["label"] = interact["term"].map(
        {
            "foul_diff_home_minus_away_before:C(playoffs)[T.1]": "Foul diff × playoffs",
            "foul_diff_home_minus_away_before:close_game": "Foul diff × close game",
            "last_foul_against_home:foul_diff_home_minus_away_before": "Last foul × foul diff",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, sub, labels, color, title in [
        (axes[0], main, [TERM_LABELS.get(t, t) for t in main["term"]], COLORS["secondary"], "Main terms"),
        (axes[1], interact, interact["label"], COLORS["home"], "Interaction terms"),
    ]:
        y_pos = np.arange(len(sub))
        ax.errorbar(
            sub["coef"],
            y_pos,
            xerr=1.96 * sub["std_err"],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linewidth=1.2,
        )
        add_zero_line(ax)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Coefficient (95% CI)")
        ax.set_title(title, pad=8)
        style_axes(ax, ygrid=False)
    fig.suptitle(
        "Sequential model with playoff, closeness, and cross terms",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    _save(fig, "publication_interaction_coefficients.png")


def plot_discrete_changes() -> None:
    path = TABLE_DIR / "publication_discrete_changes.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    labels = ["Prev. foul on away", "Prev. foul on home"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(
        labels,
        df["discrete_change"],
        color=[COLORS["away"], COLORS["home"]],
        edgecolor="white",
        linewidth=1.0,
        width=0.58,
    )
    add_zero_line(ax)
    ax.set_ylabel("ΔP(next foul against home)")
    ax.set_title("Discrete change when game foul diff increases by 1", pad=10)
    for idx, (_, row) in enumerate(df.iterrows()):
        y = row["discrete_change"]
        va = "bottom" if y >= 0 else "top"
        offset = 0.0015 if y >= 0 else -0.0015
        ax.text(idx, y + offset, f"{y:.3f}", ha="center", va=va, fontsize=9, color=COLORS["primary"])
    style_axes(ax)
    _save(fig, "publication_discrete_changes.png")


def _plot_prob_heatmap(matrix: pd.DataFrame, row_labels: list[str], *, title: str, name: str, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    im = ax.imshow(matrix.values, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([int(c) for c in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Game foul diff before event")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.47 or val > 0.53 else COLORS["primary"]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("P(next foul against home)", rotation=90, va="center")
    ax.set_title(title, pad=10)
    style_axes(ax, ygrid=False)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    _save(fig, name)


def plot_predicted_prob_heatmap() -> None:
    path = TABLE_DIR / "publication_predicted_probabilities.csv"
    if not path.exists():
        return
    pred = pd.read_csv(path)
    matrix = pred.pivot(
        index="last_foul_against_home",
        columns="foul_diff_home_minus_away_before",
        values="predicted_p_foul_against_home",
    )
    _plot_prob_heatmap(
        matrix,
        ["Prev. foul on away", "Prev. foul on home"],
        title="Model-predicted call direction grid",
        name="publication_predicted_prob_heatmap.png",
        vmin=0.45,
        vmax=0.55,
    )


def plot_foul_diff_last_foul_heatmap(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample["foul_diff_bin"] = sample["foul_diff_home_minus_away_before"].clip(-4, 4)
    matrix = (
        sample.groupby(["last_foul_against_home", "foul_diff_bin"], observed=True)["foul_against_home"]
        .mean()
        .unstack()
    )
    _plot_prob_heatmap(
        matrix,
        ["Prev. foul on away", "Prev. foul on home"],
        title="Raw call direction by foul diff and previous call",
        name="publication_foul_diff_last_foul_heatmap.png",
        vmin=0.35,
        vmax=0.65,
    )


def plot_last_foul_by_time(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample = sample[sample["time_since_last_foul"].notna()]
    sample["time_bin"] = pd.cut(
        sample["time_since_last_foul"],
        bins=[0, 30, 60, 120, 300, 720],
        labels=["0–30s", "31–60s", "1–2m", "2–5m", "5m+"],
    )
    rows = []
    for time_bin in sample["time_bin"].cat.categories:
        sub = sample[sample["time_bin"] == time_bin]
        for last in (0, 1):
            vals = sub.loc[sub["last_foul_against_home"] == last, "foul_against_home"]
            p = vals.mean()
            n = len(vals)
            lo, hi = wilson_interval(p * n, n)
            rows.append(
                {
                    "time_bin": time_bin,
                    "last_foul_against_home": last,
                    "p": p,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    width = 0.36
    bins = list(plot_df["time_bin"].unique())
    x = np.arange(len(bins))
    away = plot_df[plot_df["last_foul_against_home"] == 0].set_index("time_bin").loc[bins]
    home = plot_df[plot_df["last_foul_against_home"] == 1].set_index("time_bin").loc[bins]
    ax.bar(x - width / 2, away["p"], width, label="Prev. foul on away", color=COLORS["away"], edgecolor="white")
    ax.bar(x + width / 2, home["p"], width, label="Prev. foul on home", color=COLORS["home"], edgecolor="white")
    for offset, sub_df in [(-width / 2, away), (width / 2, home)]:
        ax.errorbar(
            x + offset,
            sub_df["p"],
            yerr=[sub_df["p"] - sub_df["ci_low"], sub_df["ci_high"] - sub_df["p"]],
            fmt="none",
            ecolor=COLORS["primary"],
            capsize=3,
            linewidth=0.9,
        )
    add_half_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=12)
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Previous call direction vs elapsed time since last foul", pad=10)
    ax.legend(loc="upper right")
    style_axes(ax)
    _save(fig, "publication_last_foul_by_time.png")


def plot_home_possession(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample = sample[sample["home_possession"].notna()]
    rows = []
    for poss in (0, 1):
        sub = sample[sample["home_possession"] == poss]
        for last in (0, 1):
            vals = sub.loc[sub["last_foul_against_home"] == last, "foul_against_home"]
            p, n = vals.mean(), len(vals)
            lo, hi = wilson_interval(p * n, n)
            rows.append({"home_possession": poss, "last_foul_against_home": last, "p": p, "ci_low": lo, "ci_high": hi})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    width = 0.36
    labels = ["Away possession", "Home possession"]
    x = np.arange(len(labels))
    away = plot_df[plot_df["last_foul_against_home"] == 0].set_index("home_possession").loc[[0, 1]]
    home = plot_df[plot_df["last_foul_against_home"] == 1].set_index("home_possession").loc[[0, 1]]
    ax.bar(x - width / 2, away["p"], width, label="Prev. foul on away", color=COLORS["away"], edgecolor="white")
    ax.bar(x + width / 2, home["p"], width, label="Prev. foul on home", color=COLORS["home"], edgecolor="white")
    add_half_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Previous call direction by possession team", pad=10)
    ax.legend(loc="upper right")
    style_axes(ax)
    _save(fig, "publication_home_possession.png")


def plot_game_clock_heterogeneity(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample["clock_bin"] = pd.qcut(
        sample["seconds_remaining_game"],
        q=4,
        labels=["Final 12 min", "12–24 min left", "24–36 min left", "First 12 min"],
    )
    palette = [COLORS["home"], COLORS["accent"], COLORS["away"], COLORS["secondary"]]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for clock_bin, color in zip(sample["clock_bin"].cat.categories, palette, strict=True):
        sub = sample[sample["clock_bin"] == clock_bin].copy()
        sub["bin"] = sub["period_foul_diff_home_minus_away_before"].clip(-3, 3)
        grouped = (
            sub.groupby("bin", observed=True)["foul_against_home"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = proportion_ci_table(grouped)
        ax.plot(
            grouped["bin"],
            grouped["mean"],
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=str(clock_bin),
            color=color,
            linewidth=1.8,
        )
    add_half_line(ax)
    ax.set_xlabel("Period foul diff bin")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Period foul diff gradient by game clock", pad=10)
    ax.legend(title="Game time remaining", fontsize=8, loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax)
    _save(fig, "publication_game_clock_heterogeneity.png")


def plot_bonus_status(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    conditions = [
        ((0, 0), "Neither"),
        ((1, 0), "Home only"),
        ((0, 1), "Away only"),
        ((1, 1), "Both"),
    ]
    labels, values, err_low, err_high = [], [], [], []
    for (home_b, away_b), label in conditions:
        sub = sample[
            (sample["home_in_bonus_before"] == home_b) & (sample["away_in_bonus_before"] == away_b)
        ]
        p = sub["foul_against_home"].mean()
        n = len(sub)
        lo, hi = wilson_interval(p * n, n)
        labels.append(label)
        values.append(p)
        err_low.append(p - lo)
        err_high.append(hi - p)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=1.0,
        width=0.62,
    )
    ax.errorbar(x, values, yerr=[err_low, err_high], fmt="none", ecolor=COLORS["primary"], capsize=4)
    add_half_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Next call direction by bonus status before event", pad=10)
    style_axes(ax)
    annotate_bars(ax, bars, values, fmt="{:.1%}", offset=0.006)
    _save(fig, "publication_bonus_status.png")


def plot_overtime_vs_regulation(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    sample["ot"] = (sample["period"] > 4).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for ax, ot, title in zip(axes, [0, 1], ["Regulation (Q1–Q4)", "Overtime"], strict=True):
        _plot_last_foul_bars(ax, sample[sample["ot"] == ot], title=title)
        ax.set_ylim(0.34, 0.66)

    fig.suptitle(
        "Previous call direction: regulation vs overtime",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    _save(fig, "publication_overtime_vs_regulation.png")


def plot_period_diff_with_ci(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    table = period_foul_diff_next_call_table(df)
    table = proportion_ci_table(table.rename(columns={"p_foul_against_home": "mean", "n_fouls": "count"}))

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.fill_between(
        table["period_foul_diff_bin"],
        table["ci_low"],
        table["ci_high"],
        color=COLORS["away"],
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(
        table["period_foul_diff_bin"],
        table["mean"],
        marker="o",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
        color=COLORS["away"],
        linewidth=2.2,
    )
    add_half_line(ax)
    ax.set_xlabel("Home minus away period fouls before event")
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Period foul differential vs next call direction", pad=10)
    style_axes(ax)
    _save(fig, "publication_period_diff_with_ci.png")


def plot_foul_type_facets(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    for ax, (query, title) in zip(axes.ravel(), FOUL_TYPE_PANELS, strict=True):
        sub = sample.query(query)
        if len(sub) < 500:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        _plot_last_foul_bars(ax, sub, title=title)
        ax.set_ylim(0.3, 0.7)
    fig.suptitle(
        "Previous call direction by foul type",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=0.98,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.22)
    _save(fig, "publication_foul_type_facets.png")


def plot_model_comparison_dual() -> None:
    seq = pd.read_csv(TABLE_DIR / "publication_clustered_main.csv")
    seq = seq[(seq["model"] == "sequential_cluster") & (seq["term"].isin(KEY_TERMS))]
    seq = seq.set_index("term").loc[KEY_TERMS].reset_index()
    fe = pd.read_csv(TABLE_DIR / "publication_game_fe_lpm.csv")
    fe = fe[fe["term"].isin(KEY_TERMS)].set_index("term").loc[KEY_TERMS].reset_index()
    fe["model"] = "game_fe_lpm"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    _plot_forest_panel(
        axes[0],
        seq,
        title="Sequential logit (cluster SE)",
        color=COLORS["secondary"],
    )
    _plot_forest_panel(
        axes[1],
        fe,
        title="Game fixed effects LPM",
        color=COLORS["primary"],
    )
    axes[0].set_yticklabels([TERM_LABELS[t] for t in KEY_TERMS])
    axes[1].set_yticklabels([])
    fig.suptitle(
        "Within-game identification: logit vs game FE",
        fontsize=13,
        fontweight="700",
        color=COLORS["primary"],
        y=1.02,
    )
    _save(fig, "publication_model_comparison_dual.png")


def plot_figure1_main_results(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    fig = plt.figure(figsize=(13.5, 10.5))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], hspace=0.34, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_last_foul_bars(ax_a, sample)
    ax_a.set_title("Previous call direction", pad=8)
    ax_a.set_ylim(0.34, 0.66)
    add_panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_binned_probability_line(
        ax_b,
        sample,
        "period_foul_diff_home_minus_away_before",
        clip=(-4, 4),
        xlabel="Period foul diff",
        title="Period foul differential",
    )
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    table = foul_diff_next_call_table(df)
    table = proportion_ci_table(table.rename(columns={"p_foul_against_home": "mean", "n_fouls": "count"}))
    ax_c.bar(
        table["foul_diff_bin"],
        table["mean"],
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=1.0,
        width=0.82,
    )
    ax_c.errorbar(
        table["foul_diff_bin"],
        table["mean"],
        yerr=[table["mean"] - table["ci_low"], table["ci_high"] - table["mean"]],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=3,
        linewidth=0.9,
    )
    add_half_line(ax_c)
    ax_c.set_xlabel("Game foul diff bin")
    ax_c.set_ylabel("P(next foul against home)")
    ax_c.set_title("Game foul differential", pad=8)
    ax_c.set_ylim(0.38, 0.62)
    style_axes(ax_c)
    add_panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    pred_path = TABLE_DIR / "publication_predicted_probabilities.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        for last_foul, label, color in [
            (0, "Prev. foul on away", COLORS["away"]),
            (1, "Prev. foul on home", COLORS["home"]),
        ]:
            sub = pred[pred["last_foul_against_home"] == last_foul]
            ax_d.plot(
                sub["foul_diff_home_minus_away_before"],
                sub["predicted_p_foul_against_home"],
                marker="o",
                markersize=5.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=label,
                color=color,
                linewidth=2.0,
            )
    add_half_line(ax_d)
    ax_d.set_xlabel("Game foul diff")
    ax_d.set_ylabel("Predicted P(next foul against home)")
    ax_d.set_title("Model-adjusted predictions", pad=8)
    ax_d.legend(loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax_d)
    add_panel_label(ax_d, "D")

    _save(fig, "publication_figure1_main_results.png")


def _season_previous_call_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive pp gap: P(home foul | prev away) minus P(home foul | prev home)."""
    sample = extended_descriptive_sample(df)
    rows: list[dict] = []
    for season in sorted(sample["season"].unique()):
        sub = sample[sample["season"] == season]
        if len(sub) < 2000:
            continue
        table = _last_foul_table(sub).sort_values("last_foul_against_home")
        if len(table) < 2:
            continue
        p_after_away = float(table.iloc[0]["p"])
        p_after_home = float(table.iloc[1]["p"])
        n_away = float(table.iloc[0]["n"])
        n_home = float(table.iloc[1]["n"])
        gap_pp = (p_after_away - p_after_home) * 100
        se_pp = (
            np.sqrt(
                p_after_away * (1 - p_after_away) / n_away
                + p_after_home * (1 - p_after_home) / n_home
            )
            * 100
        )
        end_year = int(season)
        rows.append(
            {
                "season": end_year,
                "season_label": f"{end_year - 1}–{str(end_year)[-2:]}",
                "gap_pp": gap_pp,
                "se_pp": se_pp,
                "n_obs": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def _plot_season_previous_call_gap_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    gaps = _season_previous_call_gaps(df)
    if gaps.empty:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        return

    pooled_gap = float(
        np.average(gaps["gap_pp"], weights=gaps["n_obs"])
    )
    y = np.arange(len(gaps))
    ax.hlines(
        y,
        0,
        gaps["gap_pp"],
        color=COLORS["secondary"],
        linewidth=2.0,
        alpha=0.55,
        zorder=2,
    )
    ax.errorbar(
        gaps["gap_pp"],
        y,
        xerr=1.96 * gaps["se_pp"],
        fmt="o",
        markersize=7,
        color=TERM_COLORS["last_foul_against_home"],
        ecolor=TERM_COLORS["last_foul_against_home"],
        markeredgecolor="white",
        markeredgewidth=0.8,
        capsize=3,
        linewidth=1.2,
        zorder=3,
    )
    ax.axvline(
        pooled_gap,
        color=COLORS["primary"],
        linewidth=1.5,
        linestyle="--",
        zorder=1,
    )
    pooled_label_x = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        pooled_gap + 0.08,
        0.96,
        f"Pooled: {pooled_gap:.1f} pp",
        transform=pooled_label_x,
        color=COLORS["primary"],
        fontsize=8.5,
        va="top",
        ha="left",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(gaps["season_label"])
    ax.set_xlabel("Probability gap (percentage points)")
    ax.set_title(
        "Descriptive previous-call gap by season",
        pad=10,
        fontsize=10,
    )
    xmax = float((gaps["gap_pp"] + 1.96 * gaps["se_pp"]).max())
    ax.set_xlim(0, max(xmax * 1.12, pooled_gap * 1.15))
    style_axes(ax, ygrid=False)
    ax.invert_yaxis()


def plot_figure2_robustness(df: pd.DataFrame) -> None:
    """Descriptive season stability, AMEs, and within-game identification."""
    fig = plt.figure(figsize=(15.0, 5.1))
    gs = GridSpec(1, 3, figure=fig, wspace=0.72, width_ratios=[1.0, 1.32, 1.08])

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_season_previous_call_gap_panel(ax_a, df)
    add_panel_label(ax_a, "A", x=-0.20, y=1.06)

    ax_b = fig.add_subplot(gs[0, 1], sharey=None)
    me = pd.read_csv(TABLE_DIR / "publication_marginal_effects.csv")
    me = me[(me["model"] == "sequential_cluster") & (me["term"].isin(FIGURE4_TERMS))]
    me = me.set_index("term").loc[FIGURE4_TERMS].reset_index()
    y_pos = np.arange(len(me))
    colors = [TERM_COLORS[t] for t in me["term"]]
    ax_b.barh(
        y_pos,
        me["dy_dx"],
        color=colors,
        edgecolor="white",
        height=0.58,
        zorder=2,
    )
    ax_b.errorbar(
        me["dy_dx"],
        y_pos,
        xerr=1.96 * me["std_err"],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=4,
        linewidth=1.0,
        zorder=3,
    )
    add_zero_line(ax_b)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([FIGURE4_FORMAL_LABELS[t] for t in me["term"]], fontsize=8.5)
    ax_b.tick_params(axis="y", pad=2)
    ax_b.set_xlabel("AME (probability points, 95% CI)", fontsize=9, labelpad=8)
    ax_b.set_title("Sequential model AMEs", pad=10, fontsize=10)
    style_axes(ax_b, ygrid=False)
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[0, 2])
    me_seq = pd.read_csv(TABLE_DIR / "publication_marginal_effects.csv")
    me_seq = me_seq[
        (me_seq["model"] == "sequential_cluster") & (me_seq["term"].isin(FIGURE4_TERMS))
    ].set_index("term")
    fe = pd.read_csv(TABLE_DIR / "publication_game_fe_lpm.csv").set_index("term")
    x = np.arange(len(FIGURE4_TERMS))
    width = 0.36
    seq_ame = [me_seq.loc[t, "dy_dx"] for t in FIGURE4_TERMS]
    seq_se = [me_seq.loc[t, "std_err"] for t in FIGURE4_TERMS]
    fe_coef = [fe.loc[t, "coef"] for t in FIGURE4_TERMS]
    fe_se = [fe.loc[t, "std_err"] for t in FIGURE4_TERMS]
    ax_c.bar(
        x - width / 2,
        seq_ame,
        width,
        yerr=1.96 * np.array(seq_se),
        label="Sequential logit (AME)",
        color=COLORS["secondary"],
        edgecolor="white",
        capsize=3,
        error_kw={"linewidth": 1.0, "ecolor": COLORS["primary"]},
    )
    ax_c.bar(
        x + width / 2,
        fe_coef,
        width,
        yerr=1.96 * np.array(fe_se),
        label="Game FE (LPM)",
        color=COLORS["primary"],
        edgecolor="white",
        capsize=3,
        error_kw={"linewidth": 1.0, "ecolor": COLORS["primary"]},
    )
    add_zero_line(ax_c)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(
        [FIGURE4_SHORT_LABELS[t] for t in FIGURE4_TERMS],
        rotation=0,
        ha="center",
        fontsize=9,
    )
    ax_c.set_ylabel("Probability points", fontsize=9, labelpad=10)
    ax_c.set_title("Estimator comparison", pad=10, fontsize=10)
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=True,
        edgecolor=COLORS["grid"],
        fontsize=8,
    )
    style_axes(ax_c)
    add_panel_label(ax_c, "C")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.84, bottom=0.24, wspace=0.72)
    _save(fig, "publication_figure2_robustness.png")


def plot_figure3_heterogeneity(df: pd.DataFrame) -> None:
    sample = extended_descriptive_sample(df)
    fig = plt.figure(figsize=(13.5, 4.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    for playoffs, label, color in [
        (0, "Regular season", COLORS["regular"]),
        (1, "Playoffs", COLORS["playoffs"]),
    ]:
        sub = sample[sample["playoffs"] == playoffs].copy()
        sub["bin"] = sub["period_foul_diff_home_minus_away_before"].clip(-3, 3)
        grouped = (
            sub.groupby("bin", observed=True)["foul_against_home"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped = proportion_ci_table(grouped)
        ax_a.plot(
            grouped["bin"],
            grouped["mean"],
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=label,
            color=color,
            linewidth=1.8,
        )
    add_half_line(ax_a)
    ax_a.set_xlabel("Period foul diff")
    ax_a.set_ylabel("P(next foul against home)")
    ax_a.set_title("Regular season vs playoffs", pad=8)
    ax_a.legend(loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax_a)
    add_panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    palette = [COLORS["secondary"], COLORS["away"], COLORS["accent"], COLORS["home"]]
    sample["margin_bin"] = pd.cut(
        sample["score_margin_home_before"].abs(),
        bins=[0, 5, 10, 20, 100],
        labels=["0–5", "6–10", "11–20", "21+"],
    )
    for margin_bin, color in zip(["0–5", "6–10", "11–20", "21+"], palette, strict=True):
        sub = sample[sample["margin_bin"] == margin_bin].copy()
        sub["bin"] = sub["period_foul_diff_home_minus_away_before"].clip(-3, 3)
        grouped = sub.groupby("bin", observed=True)["foul_against_home"].mean()
        ax_b.plot(grouped.index, grouped.values, marker="o", markersize=4.5, label=f"|margin| {margin_bin}", color=color)
    add_half_line(ax_b)
    ax_b.set_xlabel("Period foul diff")
    ax_b.set_ylabel("P(next foul against home)")
    ax_b.set_title("By score margin", pad=8)
    ax_b.legend(title="Score margin", fontsize=8, loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax_b)
    add_panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[0, 2])
    sample_ot = sample.copy()
    sample_ot["segment"] = np.where(sample_ot["period"] > 4, "Overtime", "Regulation")
    for segment, color in [("Regulation", COLORS["regular"]), ("Overtime", COLORS["playoffs"])]:
        sub = sample_ot[sample_ot["segment"] == segment]
        table = _last_foul_table(sub)
        table = table.sort_values("last_foul_against_home")
        offset = -0.18 if segment == "Regulation" else 0.18
        x = table["last_foul_against_home"] + offset
        ax_c.bar(
            x,
            table["p"],
            width=0.32,
            label=segment,
            color=color,
            edgecolor="white",
            alpha=0.95,
        )
    add_half_line(ax_c)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["Prev. foul on away", "Prev. foul on home"])
    ax_c.set_ylabel("P(next foul against home)")
    ax_c.set_title("Regulation vs overtime", pad=8)
    ax_c.legend(loc="upper right", frameon=True, edgecolor=COLORS["grid"])
    style_axes(ax_c)
    add_panel_label(ax_c, "C")

    _save(fig, "publication_figure3_heterogeneity.png")


def _draw_flow_box(
    ax: plt.Axes,
    center: tuple[float, float],
    *,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    facecolor: str,
    edgecolor: str | None = None,
    linestyle: str = "-",
) -> None:
    x, y = center
    edge = edgecolor or facecolor
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.5,
        edgecolor=edge,
        facecolor=facecolor,
        alpha=0.22 if linestyle == "--" else 0.32,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(
            x,
            y + 0.11,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="600",
            color=COLORS["primary"],
            zorder=3,
        )
        ax.text(
            x,
            y - 0.16,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.5,
            color=COLORS["neutral"],
            zorder=3,
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="600",
            color=COLORS["primary"],
            zorder=3,
        )


def _draw_flow_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linestyle: str = "-",
    color: str | None = None,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.25,
        color=color or COLORS["primary"],
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        zorder=1,
    )
    ax.add_patch(arrow)


def _draw_flow_note(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    *,
    color: str | None = None,
    ha: str = "center",
) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        ha=ha,
        va="center",
        fontsize=7.5,
        color=color or COLORS["neutral"],
        style="italic",
        zorder=4,
    )


def plot_sample_flow(df: pd.DataFrame) -> None:
    """Sample-construction flowchart with observed N (main vs heterogeneity paths)."""
    n_all = len(df)
    n_games = int(df["game_id"].nunique())
    n_no_tf = len(df[(df["technical_foul"] == 0) & (df["flagrant_foul"] == 0)])
    n_baseline = len(baseline_sample(df))
    n_seq = len(extended_sample(df))
    n_off = int((df["offensive_foul"] == 1).sum())
    n_drop_tf = n_all - n_no_tf
    n_drop_off = n_no_tf - n_baseline
    n_drop_seq = n_baseline - n_seq

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(1.6, 9.6)
    ax.axis("off")

    cx = 5.0
    lx = 3.15
    rx = 6.85
    w_top = 5.4
    w_branch = 3.35
    h = 0.88

    y_all = 8.85
    y_tf = 7.35
    y_branch = 5.75
    y_miss = 4.15
    y_seq = 2.55

    _draw_flow_box(
        ax,
        (cx, y_all),
        width=w_top,
        height=h,
        title="Play-by-play foul events",
        subtitle=f"{n_all:,} fouls  ·  {n_games:,} games",
        facecolor=COLORS["neutral"],
    )
    _draw_flow_arrow(ax, (cx, y_all - h / 2 - 0.03), (cx, y_tf + h / 2 + 0.03))
    _draw_flow_note(ax, (cx + 2.65, (y_all + y_tf) / 2), f"−{n_drop_tf:,} technical/flagrant", ha="left")

    _draw_flow_box(
        ax,
        (cx, y_tf),
        width=w_top,
        height=h,
        title="Exclude technical and flagrant fouls",
        subtitle=f"{n_no_tf:,} fouls remaining",
        facecolor=COLORS["secondary"],
    )

    junction_y = y_tf - h / 2 - 0.18
    _draw_flow_arrow(ax, (cx, y_tf - h / 2 - 0.03), (cx, junction_y + 0.02))
    _draw_flow_arrow(
        ax,
        (cx, junction_y),
        (lx, y_branch + h / 2 + 0.03),
        connectionstyle="arc3,rad=0.12",
    )
    _draw_flow_arrow(
        ax,
        (cx, junction_y),
        (rx, y_branch + h / 2 + 0.03),
        linestyle="--",
        color=COLORS["playoffs"],
        connectionstyle="arc3,rad=-0.12",
    )

    ax.text(lx, y_branch + h / 2 + 0.42, "Main analysis", fontsize=8, fontweight="600", ha="center", color=COLORS["primary"])
    ax.text(rx, y_branch + h / 2 + 0.42, "Heterogeneity only", fontsize=8, fontweight="600", ha="center", color=COLORS["playoffs"])

    _draw_flow_box(
        ax,
        (lx, y_branch),
        width=w_branch,
        height=h + 0.08,
        title="Exclude offensive fouls",
        subtitle=f"{n_baseline:,} fouls  ·  baseline",
        facecolor=COLORS["secondary"],
    )
    _draw_flow_box(
        ax,
        (rx, y_branch),
        width=w_branch,
        height=h + 0.08,
        title="Offensive fouls retained",
        subtitle=f"{n_off:,} fouls  ·  Table 6",
        facecolor=COLORS["playoffs"],
        edgecolor=COLORS["playoffs"],
        linestyle="--",
    )
    _draw_flow_note(
        ax,
        (lx - w_branch / 2 - 0.15, y_branch + 0.02),
        f"−{n_drop_off:,} offensive",
        ha="right",
    )

    _draw_flow_arrow(ax, (lx, y_branch - (h + 0.08) / 2 - 0.03), (lx, y_miss + h / 2 + 0.03))
    _draw_flow_box(
        ax,
        (lx, y_miss),
        width=w_branch,
        height=h + 0.06,
        title="Require sequential state",
        subtitle="non-missing last foul & period diff",
        facecolor=COLORS["primary"],
    )
    _draw_flow_note(
        ax,
        (lx + w_branch / 2 + 0.12, (y_branch + y_miss) / 2),
        f"−{n_drop_seq:,} missing\nsequential state",
        ha="left",
    )

    _draw_flow_arrow(ax, (lx, y_miss - (h + 0.06) / 2 - 0.03), (lx, y_seq + h / 2 + 0.03))
    _draw_flow_box(
        ax,
        (lx, y_seq),
        width=w_branch,
        height=h + 0.06,
        title="Sequential regression sample",
        subtitle=f"{n_seq:,} fouls  ·  main tables",
        facecolor=COLORS["primary"],
    )

    ax.text(
        0.35,
        9.45,
        "Sample construction (2017–18 through 2024–25)",
        fontsize=11,
        fontweight="600",
        color=COLORS["primary"],
        ha="left",
    )
    _save(fig, "publication_sample_flow.png")


def write_all_publication_figures(df: pd.DataFrame) -> None:
    ensure_dir(FIGURE_DIR)

    plot_sample_flow(df)

    plot_coefficient_forest()
    plot_placebo_last_foul()
    plot_placebo_all_terms()
    plot_foul_type_heatmap()
    plot_marginal_effects()
    plot_interaction_coefficients()
    plot_discrete_changes()
    plot_predicted_prob_heatmap()
    plot_playoffs_vs_regular(df)
    plot_period_heterogeneity(df)
    plot_score_margin_heterogeneity(df)
    plot_bonus_threshold(df)
    plot_bonus_status(df)
    plot_time_since_last_foul(df)
    plot_last_foul_by_time(df)
    plot_home_possession(df)
    plot_game_clock_heterogeneity(df)
    plot_foul_diff_last_foul_heatmap(df)
    plot_overtime_vs_regulation(df)
    plot_period_diff_with_ci(df)
    plot_foul_type_facets(df)
    plot_model_comparison_dual()

    print("Fitting sequential model for descriptive vs model figure...")
    sample = extended_sample(df)
    sequential_model = fit_logit_clustered(SEQUENTIAL_FORMULA, sample)
    plot_descriptive_vs_model(df, sequential_model)

    print("Fitting per-season models for season coefficient plot...")
    plot_season_coefficients(df)

    print("Building composite publication figures...")
    plot_figure_placebo()
    plot_figure_foul_type_coef()
    plot_figure1_main_results(df)
    plot_figure2_robustness(df)
    plot_figure3_heterogeneity(df)

    print("Wrote publication figures to outputs/figures/")
