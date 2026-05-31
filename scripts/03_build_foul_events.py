import pandas as pd

from whistle_balance.config import INTERIM_DIR, PROCESSED_DIR
from whistle_balance.data_utils import ensure_dir, write_csv
from whistle_balance.foul_parser import build_foul_events_from_pbp


def main() -> None:
    input_path = INTERIM_DIR / "nba_pbp_clean.csv"
    output_path = PROCESSED_DIR / "foul_events.csv"
    ensure_dir(PROCESSED_DIR)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}. Run scripts/02_clean_pbp.py first.")
    pbp = pd.read_csv(input_path)
    fouls = build_foul_events_from_pbp(pbp)
    write_csv(fouls, output_path)
    print(
        f"Wrote data/processed/foul_events.csv "
        f"({len(fouls):,} foul events, {fouls['game_id'].nunique():,} games)"
    )


if __name__ == "__main__":
    main()
