"""Helpers for parsing NBA play-by-play fields."""

from __future__ import annotations

import re

import pandas as pd

PT_CLOCK_PATTERN = re.compile(r"PT(?:(?P<minutes>\d+)M)?(?P<seconds>[\d.]+)S")

REGULATION_PERIOD_SECONDS = 12 * 60
OVERTIME_PERIOD_SECONDS = 5 * 60


def parse_clock_seconds(clock: str | float | int | None) -> float | None:
    """Parse PBP clock strings (PT11M18.00S, MM:SS, or SS.s)."""
    if clock is None or (isinstance(clock, float) and pd.isna(clock)):
        return None
    text = str(clock).strip()
    if not text:
        return None
    if text.startswith("PT"):
        match = PT_CLOCK_PATTERN.fullmatch(text)
        if match:
            minutes = int(match.group("minutes") or 0)
            seconds = float(match.group("seconds"))
            return minutes * 60 + seconds
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return int(minutes) * 60 + float(seconds)
    return float(text)


def seconds_remaining_in_game(period: int, seconds_remaining_period: float | None) -> float | None:
    if seconds_remaining_period is None:
        return None
    if period <= 4:
        return (4 - period) * REGULATION_PERIOD_SECONDS + seconds_remaining_period
    return seconds_remaining_period


def game_seconds_elapsed(period: int, seconds_remaining_period: float | None) -> float | None:
    if seconds_remaining_period is None:
        return None
    period_length = REGULATION_PERIOD_SECONDS if period <= 4 else OVERTIME_PERIOD_SECONDS
    if period <= 4:
        completed = (period - 1) * REGULATION_PERIOD_SECONDS
    else:
        completed = 4 * REGULATION_PERIOD_SECONDS + (period - 5) * OVERTIME_PERIOD_SECONDS
    elapsed_in_period = period_length - seconds_remaining_period
    return completed + elapsed_in_period


def parse_score(score: str | float | int | None) -> tuple[float | None, float | None]:
    """Parse SCORE field formatted as 'away - home'."""
    if score is None or (isinstance(score, float) and pd.isna(score)):
        return None, None
    text = str(score).strip()
    if not text or "-" not in text:
        return None, None
    away_text, home_text = text.split("-", 1)
    return float(home_text.strip()), float(away_text.strip())


def event_description(row: pd.Series) -> str:
    for column in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION", "description"):
        if column not in row.index:
            continue
        value = row[column]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def other_team(team_id: int | float, home_team_id: int, away_team_id: int) -> int:
    if int(team_id) == int(home_team_id):
        return int(away_team_id)
    return int(home_team_id)


ACTION_TYPE_TO_EVENT_CODE = {
    "Made Shot": 1,
    "Missed Shot": 2,
    "Free Throw": 3,
    "Rebound": 4,
    "Turnover": 5,
    "Foul": 6,
    "Jump Ball": 10,
}


def action_type_to_event_code(action_type: str | None) -> int | None:
    if not isinstance(action_type, str):
        return None
    return ACTION_TYPE_TO_EVENT_CODE.get(action_type)


def update_possession_team(
    event_msg_type: int | None,
    player1_team_id: int | float | None,
    home_team_id: int,
    away_team_id: int,
    current_team: int | None,
) -> int | None:
    """Update running possession using common PBP event rules."""
    if event_msg_type is None or pd.isna(player1_team_id):
        return current_team

    team_id = int(player1_team_id)
    if event_msg_type == 1:
        return other_team(team_id, home_team_id, away_team_id)
    if event_msg_type == 4:
        return team_id
    if event_msg_type == 5:
        return other_team(team_id, home_team_id, away_team_id)
    if event_msg_type == 10:
        return team_id
    return current_team
