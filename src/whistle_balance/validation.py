"""Validate parsed foul counts against official box scores."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from whistle_balance.config import PROCESSED_DIR, RAW_DIR
from whistle_balance.data_utils import normalize_game_id


def _official_team_fouls(box_path: Path, home_team_id: int, away_team_id: int) -> tuple[int, int]:
    box = pd.read_csv(box_path)
    home_fouls = int(box.loc[box["TEAM_ID"] == home_team_id, "PF"].iloc[0])
    away_fouls = int(box.loc[box["TEAM_ID"] == away_team_id, "PF"].iloc[0])
    return home_fouls, away_fouls


def validate_games(
    foul_events: pd.DataFrame,
    games: pd.DataFrame,
    raw_dir: Path,
    *,
    sample_size: int = 10,
) -> pd.DataFrame:
    box_dir = raw_dir / "boxscores"
    records: list[dict] = []

    sample_games = games.head(sample_size)
    for _, game in sample_games.iterrows():
        game_id = normalize_game_id(game["game_id"])
        game_fouls = foul_events[foul_events["game_id"] == game_id]
        countable = game_fouls[
            (game_fouls["technical_foul"] == 0) & (game_fouls["flagrant_foul"] == 0)
        ]
        parsed_home = int((countable["foul_against_home"] == 1).sum())
        parsed_away = int((countable["foul_against_home"] == 0).sum())
        box_path = box_dir / f"box_{game_id}.csv"
        official_home, official_away = _official_team_fouls(
            box_path,
            int(game["home_team_id"]),
            int(game["away_team_id"]),
        )
        records.append(
            {
                "game_id": game_id,
                "season": game["season"],
                "game_date": game["game_date"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "parsed_home_fouls": parsed_home,
                "parsed_away_fouls": parsed_away,
                "official_home_fouls": official_home,
                "official_away_fouls": official_away,
                "home_diff": parsed_home - official_home,
                "away_diff": parsed_away - official_away,
                "match": parsed_home == official_home and parsed_away == official_away,
            }
        )
    return pd.DataFrame(records)


def write_validation_log(results: pd.DataFrame, path: Path) -> None:
    lines = ["# Validation Log", ""]
    for row in results.itertuples(index=False):
        lines.extend(
            [
                f"### Game ID: {row.game_id}",
                f"- Season: {row.season}",
                f"- Date: {row.game_date}",
                f"- Teams: {row.away_team} @ {row.home_team}",
                f"- Parsed home fouls: {row.parsed_home_fouls}",
                f"- Parsed away fouls: {row.parsed_away_fouls}",
                f"- Official home fouls: {row.official_home_fouls}",
                f"- Official away fouls: {row.official_away_fouls}",
                f"- Home diff: {row.home_diff}",
                f"- Away diff: {row.away_diff}",
                f"- Match: {row.match}",
                "- Issues:",
                "- Notes:",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(sample_size: int = 10) -> None:
    foul_path = PROCESSED_DIR / "foul_events.csv"
    foul_events = pd.read_csv(foul_path, dtype={"game_id": str})
    foul_events["game_id"] = foul_events["game_id"].map(normalize_game_id)
    games = pd.read_csv(sorted(RAW_DIR.glob("games_*.csv"))[-1], dtype={"game_id": str})
    games["game_id"] = games["game_id"].map(normalize_game_id)
    results = validate_games(foul_events, games, RAW_DIR, sample_size=sample_size)
    out_table = Path("outputs/tables/foul_validation_sample.csv")
    out_table.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_table, index=False)
    write_validation_log(results, Path("docs/validation_log.md"))
    matched = int(results["match"].sum())
    print(f"Validated {len(results)} games: {matched}/{len(results)} exact matches.")
    print(f"Wrote {out_table} and docs/validation_log.md")
