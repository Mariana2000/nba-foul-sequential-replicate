"""Fetch NBA game lists, play-by-play, and box scores via nba_api."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, leaguegamefinder, playbyplayv3

from whistle_balance.data_utils import normalize_game_id

REQUEST_SLEEP_SECONDS = 0.6
MAX_RETRIES = 5


def season_string(season_end_year: int) -> str:
    """Convert season end year (e.g. 2024) to nba_api season string (e.g. 2023-24)."""
    return f"{season_end_year - 1}-{str(season_end_year)[-2:]}"


def _retry_call(func, *args, **kwargs):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retry on any API failure
            last_error = exc
            time.sleep(REQUEST_SLEEP_SECONDS * (attempt + 2))
    raise last_error


def parse_matchup_teams(matchup: str) -> tuple[str, str]:
    """Return (home_team_abbr, away_team_abbr) from a MATCHUP string."""
    matchup = matchup.strip()
    if " @ " in matchup:
        away, home = matchup.split(" @ ", 1)
        return home.strip(), away.strip()
    if " vs. " in matchup:
        home, away = matchup.split(" vs. ", 1)
        return home.strip(), away.strip()
    raise ValueError(f"Unrecognized MATCHUP format: {matchup!r}")


def fetch_season_games(season_end_year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """Return one row per game with home/away teams and metadata."""
    finder = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=season_string(season_end_year),
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    raw = finder.get_data_frames()[0]
    if raw.empty:
        return raw

    records: list[dict] = []
    for game_id, group in raw.groupby("GAME_ID"):
        row = group.iloc[0]
        home_team, away_team = parse_matchup_teams(row["MATCHUP"])
        team_ids = dict(zip(group["TEAM_ABBREVIATION"], group["TEAM_ID"], strict=False))
        records.append(
            {
                "game_id": normalize_game_id(game_id),
                "season": season_end_year,
                "game_date": row["GAME_DATE"],
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": team_ids[home_team],
                "away_team_id": team_ids[away_team],
                "season_type": season_type,
                "playoffs": int(season_type == "Playoffs"),
            }
        )
    return pd.DataFrame(records).sort_values(["game_date", "game_id"]).reset_index(drop=True)


def fetch_play_by_play(game_id: str) -> pd.DataFrame:
    def _call():
        return playbyplayv3.PlayByPlayV3(game_id=game_id).get_data_frames()[0]

    return _retry_call(_call)


def fetch_box_score_teams(game_id: str) -> pd.DataFrame:
    def _call():
        return boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id).get_data_frames()[1]

    return _retry_call(_call)


def download_game_bundle(
    game: pd.Series,
    pbp_dir: Path,
    box_dir: Path,
    *,
    skip_existing: bool = True,
) -> dict[str, str]:
    """Download PBP and box score for one game. Returns status metadata."""
    game_id = normalize_game_id(game["game_id"])
    pbp_path = pbp_dir / f"pbp_{game_id}.csv"
    box_path = box_dir / f"box_{game_id}.csv"

    if skip_existing and pbp_path.exists() and box_path.exists():
        return {"game_id": game_id, "status": "skipped"}

    pbp = fetch_play_by_play(game_id)
    time.sleep(REQUEST_SLEEP_SECONDS)
    box = fetch_box_score_teams(game_id)
    time.sleep(REQUEST_SLEEP_SECONDS)

    pbp.to_csv(pbp_path, index=False)
    box.to_csv(box_path, index=False)
    return {"game_id": game_id, "status": "downloaded"}


def download_season(
    season_end_year: int,
    raw_dir: Path,
    *,
    season_types: list[str] | None = None,
    max_games: int | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Download games, PBP, and box scores for a season."""
    season_types = season_types or ["Regular Season"]
    pbp_dir = raw_dir / "pbp"
    box_dir = raw_dir / "boxscores"
    pbp_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)

    frames = [fetch_season_games(season_end_year, st) for st in season_types]
    games = pd.concat(frames, ignore_index=True)
    if max_games is not None:
        games = games.head(max_games).copy()

    games_path = raw_dir / f"games_{season_end_year}.csv"
    games.to_csv(games_path, index=False)

    statuses: list[dict[str, str]] = []
    for _, game in games.iterrows():
        statuses.append(
            download_game_bundle(game, pbp_dir, box_dir, skip_existing=skip_existing)
        )

    status_df = pd.DataFrame(statuses)
    status_df.to_csv(raw_dir / f"download_status_{season_end_year}.csv", index=False)
    return games
