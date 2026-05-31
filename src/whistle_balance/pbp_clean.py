"""Standardize raw nba_api play-by-play files into one interim table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from whistle_balance.data_utils import normalize_game_id
from whistle_balance.pbp_utils import (
    action_type_to_event_code,
    parse_clock_seconds,
    seconds_remaining_in_game,
)


def _load_games(raw_dir: Path, seasons: list[int] | None = None) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("games_*.csv"))
    if seasons is not None:
        allowed = {str(s) for s in seasons}
        paths = [p for p in paths if p.stem.replace("games_", "") in allowed]
    if not paths:
        raise FileNotFoundError(f"No games_*.csv files found in {raw_dir}")
    return pd.concat((pd.read_csv(path, dtype={"game_id": str}) for path in paths), ignore_index=True)


def _coerce_score(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("", pd.NA), errors="coerce")


def _standardize_pbp_file(path: Path, game: pd.Series) -> pd.DataFrame:
    pbp = pd.read_csv(path)
    pbp = pbp.rename(
        columns={
            "gameId": "game_id",
            "actionNumber": "event_num",
            "actionType": "action_type",
            "subType": "action_sub_type",
            "period": "period",
            "clock": "clock",
            "teamId": "player1_team_id",
            "personId": "player1_id",
            "playerName": "player1_name",
            "scoreHome": "score_home_raw",
            "scoreAway": "score_away_raw",
        }
    )
    pbp["game_id"] = normalize_game_id(game["game_id"])
    pbp["season"] = game["season"]
    pbp["game_date"] = game["game_date"]
    pbp["home_team"] = game["home_team"]
    pbp["away_team"] = game["away_team"]
    pbp["home_team_id"] = game["home_team_id"]
    pbp["away_team_id"] = game["away_team_id"]
    pbp["playoffs"] = game["playoffs"]
    pbp["description"] = pbp["description"].fillna("")
    pbp["event_msg_type"] = pbp["action_type"].map(action_type_to_event_code)
    pbp["event_msg_action_type"] = pbp["action_sub_type"]
    pbp["seconds_remaining_period"] = pbp["clock"].map(parse_clock_seconds)
    pbp["seconds_remaining_game"] = pbp.apply(
        lambda row: seconds_remaining_in_game(row["period"], row["seconds_remaining_period"]),
        axis=1,
    )
    pbp["score_home"] = _coerce_score(pbp["score_home_raw"]).ffill().fillna(0)
    pbp["score_away"] = _coerce_score(pbp["score_away_raw"]).ffill().fillna(0)
    return pbp


def build_clean_pbp(raw_dir: Path, seasons: list[int] | None = None) -> pd.DataFrame:
    games = _load_games(raw_dir, seasons=seasons)
    pbp_dir = raw_dir / "pbp"
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for _, game in games.iterrows():
        path = pbp_dir / f"pbp_{normalize_game_id(game['game_id'])}.csv"
        if not path.exists():
            missing.append(str(game["game_id"]))
            continue
        frames.append(_standardize_pbp_file(path, game))

    if not frames:
        raise FileNotFoundError(
            f"No PBP files found under {pbp_dir}. Run scripts/01_download_nba_pbp.py first."
        )

    if missing:
        print(
            f"Warning: skipped {len(missing)} games without PBP files "
            f"(processed {len(frames)} games)."
        )

    clean = pd.concat(frames, ignore_index=True)
    clean = clean.sort_values(["game_id", "event_num"]).reset_index(drop=True)
    return clean
