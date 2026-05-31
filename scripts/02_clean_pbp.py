import argparse

from whistle_balance.config import INTERIM_DIR, RAW_DIR
from whistle_balance.data_utils import ensure_dir, write_csv
from whistle_balance.pbp_clean import build_clean_pbp


def main(seasons: list[int] | None) -> None:
    ensure_dir(INTERIM_DIR)
    clean = build_clean_pbp(RAW_DIR, seasons=seasons)
    output_path = INTERIM_DIR / "nba_pbp_clean.csv"
    write_csv(clean, output_path)
    print(
        f"Wrote data/interim/nba_pbp_clean.csv "
        f"({len(clean):,} events, {clean['game_id'].nunique():,} games)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize raw PBP into one interim file.")
    parser.add_argument(
        "--season",
        type=int,
        action="append",
        dest="seasons",
        help="Season end year to include (repeatable). Defaults to all downloaded seasons.",
    )
    args = parser.parse_args()
    main(args.seasons)
