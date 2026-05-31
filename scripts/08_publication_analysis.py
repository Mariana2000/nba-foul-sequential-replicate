import pandas as pd

from whistle_balance.config import PROCESSED_DIR
from whistle_balance.publication import run_publication_analysis


def main() -> None:
    path = PROCESSED_DIR / "foul_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/03_build_foul_events.py first.")
    df = pd.read_csv(path, dtype={"game_id": str})
    run_publication_analysis(df)


if __name__ == "__main__":
    main()
