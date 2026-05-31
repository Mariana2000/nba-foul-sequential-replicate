import pandas as pd

from whistle_balance.config import PROCESSED_DIR, TABLE_DIR
from whistle_balance.data_utils import ensure_dir
from whistle_balance.modeling import run_robustness_checks


ROBUSTNESS_SPECS = {
    "full_sample": "",
    "exclude_late_game": "seconds_remaining_game > 120",
    "close_games_only": "score_margin_home_before.abs() <= 10",
    "shooting_fouls_only": "shooting_foul == 1",
    "non_shooting_fouls_only": "shooting_foul == 0",
    "regular_season_only": "playoffs == 0",
    "exclude_overtime": "period <= 4",
    "playoffs_only": "playoffs == 1",
}


def main() -> None:
    path = PROCESSED_DIR / "foul_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/03_build_foul_events.py first.")
    df = pd.read_csv(path, dtype={"game_id": str})
    ensure_dir(TABLE_DIR)

    results = run_robustness_checks(df, ROBUSTNESS_SPECS)
    results.to_csv(TABLE_DIR / "robustness_extended_summary.csv", index=False)

    legacy = results[results["model"] == "baseline"][
        ["spec", "n_obs", "coef_foul_diff_home_minus_away_before", "p_foul_diff_home_minus_away_before", "status"]
    ].rename(
        columns={
            "coef_foul_diff_home_minus_away_before": "coef_foul_diff",
            "p_foul_diff_home_minus_away_before": "p_value",
        }
    )
    legacy.to_csv(TABLE_DIR / "robustness_summary.csv", index=False)

    print("Wrote outputs/tables/robustness_extended_summary.csv")
    print("Wrote outputs/tables/robustness_summary.csv")
    ok = results[results["status"] == "ok"]
    print(
        ok.pivot_table(
            index="spec",
            columns="model",
            values="coef_foul_diff_home_minus_away_before",
        ).to_string()
    )


if __name__ == "__main__":
    main()
