"""Descriptive summaries and figures for foul-call dynamics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whistle_balance.config import FIGURE_DIR, TABLE_DIR
from whistle_balance.data_utils import ensure_dir
from whistle_balance.plot_style import (
    COLORS,
    add_half_line,
    annotate_bars,
    apply_plot_style,
    proportion_ci_table,
    save_figure,
    style_axes,
    wilson_interval,
)

apply_plot_style()


def baseline_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Main descriptive sample: exclude technical, flagrant, and offensive fouls."""
    return df[
        (df["technical_foul"] == 0) & (df["flagrant_foul"] == 0) & (df["offensive_foul"] == 0)
    ].copy()


def extended_descriptive_sample(df: pd.DataFrame) -> pd.DataFrame:
    sample = baseline_sample(df)
    return sample.dropna(subset=["last_foul_against_home", "period_foul_diff_home_minus_away_before"])


def foul_diff_next_call_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of next fouls against home by game foul-differential bin."""
    sample = baseline_sample(df)
    sample["foul_diff_bin"] = sample["foul_diff_home_minus_away_before"].clip(-8, 8)
    grouped = (
        sample.groupby("foul_diff_bin", observed=True)["foul_against_home"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "p_foul_against_home", "count": "n_fouls"})
    )
    return grouped


def period_foul_diff_next_call_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of next fouls against home by period foul-differential bin."""
    sample = extended_descriptive_sample(df)
    sample["period_foul_diff_bin"] = sample["period_foul_diff_home_minus_away_before"].clip(-5, 5)
    grouped = (
        sample.groupby("period_foul_diff_bin", observed=True)["foul_against_home"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "p_foul_against_home", "count": "n_fouls"})
    )
    return grouped


def last_foul_next_call_table(df: pd.DataFrame) -> pd.DataFrame:
    """Share of next fouls against home by previous call direction."""
    sample = extended_descriptive_sample(df)
    grouped = (
        sample.groupby("last_foul_against_home", observed=True)["foul_against_home"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "p_foul_against_home", "count": "n_fouls"})
    )
    grouped["last_foul_against_home"] = grouped["last_foul_against_home"].astype(int)
    grouped["label"] = grouped["last_foul_against_home"].map(
        {0: "Previous foul on away", 1: "Previous foul on home"}
    )
    return grouped


def _plot_binned_bar(
    table: pd.DataFrame,
    x_col: str,
    path: Path,
    *,
    xlabel: str,
    title: str,
    color: str = COLORS["secondary"],
) -> None:
    styled = proportion_ci_table(
        table.rename(columns={"p_foul_against_home": "mean", "n_fouls": "count"})
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(
        styled[x_col],
        styled["mean"],
        color=color,
        edgecolor="white",
        linewidth=1.0,
        width=0.82,
        zorder=2,
    )
    ax.errorbar(
        styled[x_col],
        styled["mean"],
        yerr=[styled["mean"] - styled["ci_low"], styled["ci_high"] - styled["mean"]],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=3,
        linewidth=0.9,
        zorder=3,
    )
    add_half_line(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P(next foul against home)")
    ax.set_title(title, pad=10)
    ax.set_ylim(0.35, 0.65)
    style_axes(ax)
    save_figure(fig, path.name)


def plot_foul_diff_vs_next_call(df: pd.DataFrame, path: Path) -> None:
    table = foul_diff_next_call_table(df)
    _plot_binned_bar(
        table,
        "foul_diff_bin",
        path,
        xlabel="Home minus away fouls before event (game total)",
        title="Game foul differential vs next call direction",
    )


def plot_period_foul_diff_vs_next_call(df: pd.DataFrame, path: Path) -> None:
    table = period_foul_diff_next_call_table(df)
    _plot_binned_bar(
        table,
        "period_foul_diff_bin",
        path,
        xlabel="Home minus away fouls before event (period)",
        title="Period foul differential vs next call direction",
    )


def plot_last_foul_vs_next_call(df: pd.DataFrame, path: Path) -> None:
    table = last_foul_next_call_table(df)
    sorted_table = table.sort_values("last_foul_against_home")
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    colors = [COLORS["away"], COLORS["home"]]
    values = sorted_table["p_foul_against_home"].values
    intervals = [
        wilson_interval(row.p_foul_against_home * row.n_fouls, row.n_fouls)
        for row in sorted_table.itertuples(index=False)
    ]
    x = np.arange(2)
    bars = ax.bar(
        sorted_table["label"],
        values,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        width=0.58,
        zorder=2,
    )
    ax.errorbar(
        x,
        values,
        yerr=[[values[i] - intervals[i][0] for i in range(2)], [intervals[i][1] - values[i] for i in range(2)]],
        fmt="none",
        ecolor=COLORS["primary"],
        capsize=4,
        linewidth=1.0,
        zorder=3,
    )
    add_half_line(ax)
    ax.set_ylabel("P(next foul against home)")
    ax.set_title("Previous call direction vs next call direction", pad=10)
    ax.set_ylim(0.35, 0.65)
    style_axes(ax)
    annotate_bars(ax, bars, values, fmt="{:.1%}", offset=0.008)
    save_figure(fig, path.name)


def write_descriptive_outputs(df: pd.DataFrame) -> None:
    ensure_dir(TABLE_DIR)
    ensure_dir(FIGURE_DIR)

    sample = baseline_sample(df)
    sample.describe(include="all").to_csv(TABLE_DIR / "foul_events_summary.csv")

    foul_diff_next_call_table(df).to_csv(TABLE_DIR / "foul_diff_next_call.csv", index=False)
    period_foul_diff_next_call_table(df).to_csv(TABLE_DIR / "period_foul_diff_next_call.csv", index=False)
    last_foul_next_call_table(df).to_csv(TABLE_DIR / "last_foul_next_call.csv", index=False)

    plot_foul_diff_vs_next_call(df, FIGURE_DIR / "foul_diff_vs_next_call.png")
    plot_period_foul_diff_vs_next_call(df, FIGURE_DIR / "period_foul_diff_vs_next_call.png")
    plot_last_foul_vs_next_call(df, FIGURE_DIR / "last_foul_vs_next_call.png")

    overall = sample["foul_against_home"].mean()
    print(f"Baseline sample: {len(sample):,} foul events across {sample['game_id'].nunique():,} games.")
    print(f"Overall P(foul against home): {overall:.3f}")
    print("Wrote outputs/tables/foul_events_summary.csv")
    print("Wrote outputs/tables/foul_diff_next_call.csv")
    print("Wrote outputs/tables/period_foul_diff_next_call.csv")
    print("Wrote outputs/tables/last_foul_next_call.csv")
    print("Wrote outputs/figures/foul_diff_vs_next_call.png")
    print("Wrote outputs/figures/period_foul_diff_vs_next_call.png")
    print("Wrote outputs/figures/last_foul_vs_next_call.png")
