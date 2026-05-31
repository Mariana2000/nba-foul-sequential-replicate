"""Shared matplotlib styling for publication figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whistle_balance.config import FIGURE_DIR
from whistle_balance.data_utils import ensure_dir

COLORS = {
    "away": "#2A9D8F",
    "home": "#E76F51",
    "primary": "#264653",
    "secondary": "#457B9D",
    "accent": "#E9C46A",
    "playoffs": "#9B2226",
    "regular": "#457B9D",
    "neutral": "#6C757D",
    "grid": "#ECECEC",
    "zero": "#888888",
    "placebo": "#C4C4C4",
    "ci_fill": "#457B9D",
}

TERM_COLORS = {
    "foul_diff_home_minus_away_before": COLORS["secondary"],
    "period_foul_diff_home_minus_away_before": COLORS["away"],
    "last_foul_against_home": COLORS["home"],
}

STYLE_APPLIED = False


def apply_plot_style() -> None:
    global STYLE_APPLIED
    if STYLE_APPLIED:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2F2F2F",
            "axes.labelcolor": "#2F2F2F",
            "axes.titlesize": 11,
            "axes.titleweight": "600",
            "axes.labelsize": 10,
            "axes.titlepad": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 220,
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        }
    )
    STYLE_APPLIED = True


def style_axes(ax: plt.Axes, *, ygrid: bool = True, xgrid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=4, color="#2F2F2F")
    if ygrid:
        ax.yaxis.grid(True, color=COLORS["grid"], linewidth=0.9, linestyle="-")
    if xgrid:
        ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.9, linestyle="-")
    ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="700",
        color=COLORS["primary"],
        va="top",
        ha="left",
    )


def add_half_line(ax: plt.Axes, y: float = 0.5) -> None:
    ax.axhline(y, color=COLORS["zero"], linestyle="--", linewidth=1.0, zorder=0)


def add_zero_line(ax: plt.Axes) -> None:
    ax.axvline(0, color=COLORS["zero"], linestyle="--", linewidth=1.0, zorder=0)


def wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)


def proportion_ci_table(
    grouped: pd.DataFrame,
    *,
    p_col: str = "mean",
    n_col: str = "count",
) -> pd.DataFrame:
    out = grouped.copy()
    successes = out[p_col] * out[n_col]
    intervals = [wilson_interval(s, n) for s, n in zip(successes, out[n_col], strict=True)]
    out["ci_low"] = [lo for lo, _ in intervals]
    out["ci_high"] = [hi for _, hi in intervals]
    out["err_low"] = out[p_col] - out["ci_low"]
    out["err_high"] = out["ci_high"] - out[p_col]
    return out


def save_figure(fig: plt.Figure, name: str, *, dpi: int = 220) -> Path:
    apply_plot_style()
    ensure_dir(FIGURE_DIR)
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def annotate_bars(ax: plt.Axes, bars, values, *, fmt: str = "{:.1%}", offset: float = 0.012) -> None:
    for bar, value in zip(bars, values, strict=True):
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=COLORS["primary"],
        )
