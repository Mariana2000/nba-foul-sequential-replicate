import argparse

from whistle_balance.config import RAW_DIR
from whistle_balance.data_utils import ensure_dir
from whistle_balance.nba_api_client import download_season


def main(
    start_season: int,
    end_season: int,
    max_games: int | None,
    include_playoffs: bool,
    force: bool,
) -> None:
    ensure_dir(RAW_DIR)
    season_types = ["Regular Season"]
    if include_playoffs:
        season_types.append("Playoffs")

    for season in range(start_season, end_season + 1):
        print(f"Downloading season ending {season}...")
        games = download_season(
            season,
            RAW_DIR,
            season_types=season_types,
            max_games=max_games,
            skip_existing=not force,
        )
        print(f"Season {season}: {len(games)} games in manifest.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NBA play-by-play and box scores.")
    parser.add_argument("--start-season", type=int, default=2024)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Limit games per season (useful for testing).",
    )
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    main(
        args.start_season,
        args.end_season,
        args.max_games,
        args.include_playoffs,
        args.force,
    )
