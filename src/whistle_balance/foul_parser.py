import pandas as pd

from whistle_balance.pbp_utils import (
    action_type_to_event_code,
    game_seconds_elapsed,
    other_team,
    update_possession_team,
)

FOUL_ACTION_TYPE = "Foul"
# NBA: opponent enters bonus starting with the 5th team foul in a quarter.
BONUS_FOUL_THRESHOLD = 5

OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "period",
    "clock",
    "seconds_remaining_game",
    "seconds_remaining_period",
    "home_team",
    "away_team",
    "team_against",
    "opponent_team",
    "foul_against_home",
    "home_possession",
    "score_home_before",
    "score_away_before",
    "score_margin_home_before",
    "home_fouls_before",
    "away_fouls_before",
    "foul_diff_home_minus_away_before",
    "home_period_fouls_before",
    "away_period_fouls_before",
    "period_foul_diff_home_minus_away_before",
    "home_in_bonus_before",
    "away_in_bonus_before",
    "last_foul_against_home",
    "time_since_last_foul",
    "foul_type",
    "shooting_foul",
    "offensive_foul",
    "loose_ball_foul",
    "technical_foul",
    "flagrant_foul",
    "playoffs",
]


def is_foul_description(desc: str) -> bool:
    if not isinstance(desc, str):
        return False
    return "foul" in desc.lower()


def classify_foul(desc: str, sub_type: str | None = None) -> dict:
    d = desc.lower() if isinstance(desc, str) else ""
    st = sub_type.lower() if isinstance(sub_type, str) else ""
    is_technical = int(
        "technical foul" in d
        or "tech foul" in d
        or "t.foul" in d
        or "technical" in st
        or "flopping" in d
        or st == "flopping"
        or "def. 3 sec" in d
        or "defense 3 second" in d
    )
    is_flagrant = int("flagrant" in d or st.startswith("flagrant"))
    return {
        "shooting_foul": int("shooting foul" in d or "s.foul" in d or st == "shooting"),
        "offensive_foul": int(
            "offensive foul" in d or "off.foul" in d or st == "offensive"
        ),
        "loose_ball_foul": int("loose ball foul" in d or "l.b.foul" in d or "loose ball" in st),
        "technical_foul": is_technical,
        "flagrant_foul": is_flagrant,
        "foul_type": _foul_type(d, st),
    }


def _foul_type(d: str, st: str = "") -> str:
    if "shooting foul" in d or st == "shooting":
        return "shooting"
    if "offensive foul" in d or "off.foul" in d or st == "offensive":
        return "offensive"
    if "loose ball foul" in d or "loose ball" in st:
        return "loose_ball"
    if "technical foul" in d or "technical" in st or "t.foul" in d or "tech foul" in d:
        return "technical"
    if "flopping" in d or st == "flopping":
        return "technical"
    if "def. 3 sec" in d or "defense 3 second" in d:
        return "technical"
    if "flagrant" in d:
        return "flagrant"
    if "personal foul" in d or st == "personal":
        return "personal"
    if "foul" in d:
        return "other"
    return "not_foul"


def _counts_toward_foul_balance(foul_flags: dict) -> bool:
    return foul_flags["technical_foul"] == 0 and foul_flags["flagrant_foul"] == 0


def _team_abbr(team_id: int, home_team_id: int, home_team: str, away_team: str) -> str:
    return home_team if int(team_id) == int(home_team_id) else away_team


def _process_game_fouls(game_df: pd.DataFrame) -> list[dict]:
    game_df = game_df.sort_values("event_num")
    home_team_id = int(game_df.iloc[0]["home_team_id"])
    away_team_id = int(game_df.iloc[0]["away_team_id"])
    home_team = game_df.iloc[0]["home_team"]
    away_team = game_df.iloc[0]["away_team"]

    home_fouls = 0
    away_fouls = 0
    home_period_fouls = 0
    away_period_fouls = 0
    current_period = None
    possession_team_id: int | None = None
    last_foul_against_home: int | None = None
    last_foul_elapsed: float | None = None
    score_home, score_away = 0.0, 0.0

    foul_events: list[dict] = []

    for row in game_df.itertuples(index=False):
        period = int(row.period)
        if current_period != period:
            home_period_fouls = 0
            away_period_fouls = 0
            current_period = period

        parsed_home, parsed_away = getattr(row, "score_home", None), getattr(row, "score_away", None)
        if parsed_home is not None and pd.notna(parsed_home):
            score_home = float(parsed_home)
        if parsed_away is not None and pd.notna(parsed_away):
            score_away = float(parsed_away)

        home_possession = None
        if possession_team_id is not None:
            home_possession = int(possession_team_id == home_team_id)

        action_type = getattr(row, "action_type", None)
        is_foul = action_type == FOUL_ACTION_TYPE or (
            pd.notna(getattr(row, "event_msg_type", None))
            and int(row.event_msg_type) == 6
        )
        if is_foul and pd.notna(row.player1_team_id):
            foul_team_id = int(row.player1_team_id)
            foul_against_home = int(foul_team_id == home_team_id)
            team_against = _team_abbr(foul_team_id, home_team_id, home_team, away_team)
            opponent_team = away_team if foul_against_home else home_team
            elapsed = game_seconds_elapsed(period, row.seconds_remaining_period)
            time_since_last_foul = None
            if last_foul_elapsed is not None and elapsed is not None:
                time_since_last_foul = elapsed - last_foul_elapsed

            foul_flags = classify_foul(row.description, getattr(row, "action_sub_type", None))
            foul_events.append(
                {
                    "game_id": row.game_id,
                    "season": row.season,
                    "game_date": row.game_date,
                    "period": period,
                    "clock": row.clock,
                    "seconds_remaining_game": row.seconds_remaining_game,
                    "seconds_remaining_period": row.seconds_remaining_period,
                    "home_team": home_team,
                    "away_team": away_team,
                    "team_against": team_against,
                    "opponent_team": opponent_team,
                    "foul_against_home": foul_against_home,
                    "home_possession": home_possession,
                    "score_home_before": score_home,
                    "score_away_before": score_away,
                    "score_margin_home_before": score_home - score_away,
                    "home_fouls_before": home_fouls,
                    "away_fouls_before": away_fouls,
                    "foul_diff_home_minus_away_before": home_fouls - away_fouls,
                    "home_period_fouls_before": home_period_fouls,
                    "away_period_fouls_before": away_period_fouls,
                    "period_foul_diff_home_minus_away_before": home_period_fouls - away_period_fouls,
                    "home_in_bonus_before": int(away_period_fouls >= BONUS_FOUL_THRESHOLD),
                    "away_in_bonus_before": int(home_period_fouls >= BONUS_FOUL_THRESHOLD),
                    "last_foul_against_home": last_foul_against_home,
                    "time_since_last_foul": time_since_last_foul,
                    "playoffs": row.playoffs,
                    **foul_flags,
                }
            )

            if _counts_toward_foul_balance(foul_flags):
                if foul_against_home:
                    home_fouls += 1
                    home_period_fouls += 1
                else:
                    away_fouls += 1
                    away_period_fouls += 1
                last_foul_against_home = foul_against_home
                if elapsed is not None:
                    last_foul_elapsed = elapsed

            if "offensive foul" in row.description.lower():
                possession_team_id = other_team(foul_team_id, home_team_id, away_team_id)

        event_code = action_type_to_event_code(action_type)
        if event_code is None and pd.notna(getattr(row, "event_msg_type", None)):
            event_code = int(row.event_msg_type)
        possession_team_id = update_possession_team(
            event_code,
            row.player1_team_id,
            home_team_id,
            away_team_id,
            possession_team_id,
        )

    return foul_events


def build_foul_events_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    required = {
        "game_id",
        "event_num",
        "period",
        "clock",
        "description",
        "player1_team_id",
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
        "season",
        "game_date",
        "playoffs",
        "seconds_remaining_period",
        "seconds_remaining_game",
        "score_home",
        "score_away",
    }
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Missing required PBP columns: {sorted(missing)}")

    events: list[dict] = []
    for _, game_df in pbp.groupby("game_id", sort=False):
        events.extend(_process_game_fouls(game_df))

    if not events:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.DataFrame(events)[OUTPUT_COLUMNS]
