import pandas as pd

from whistle_balance.config import PROCESSED_DIR
from whistle_balance.figures import write_all_publication_figures


def main() -> None:
    path = PROCESSED_DIR / "foul_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/03_build_foul_events.py first.")
    df = pd.read_csv(path, dtype={"game_id": str})
    write_all_publication_figures(df)
    print("Done. See outputs/figures/publication_*.png")


if __name__ == "__main__":
    main()
