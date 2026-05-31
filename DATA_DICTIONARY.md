# Data Dictionary

## `data/processed/foul_events.csv`

One row per foul event.

| Column | Description |
|---|---|
| `game_id` | Unique game identifier |
| `season` | Season end year |
| `game_date` | Game date |
| `period` | Quarter or overtime period |
| `clock` | Clock shown in play-by-play |
| `seconds_remaining_period` | Seconds left in current period |
| `seconds_remaining_game` | Approximate seconds left in game |
| `home_team` | Home team abbreviation |
| `away_team` | Away team abbreviation |
| `team_against` | Team the foul was called against |
| `opponent_team` | Opposing team |
| `foul_against_home` | 1 if foul called against home |
| `home_possession` | 1 if home had possession before event |
| `score_home_before` | Home score before event |
| `score_away_before` | Away score before event |
| `score_margin_home_before` | Home score minus away score |
| `home_fouls_before` | Previous fouls against home |
| `away_fouls_before` | Previous fouls against away |
| `foul_diff_home_minus_away_before` | Home previous fouls minus away previous fouls |
| `home_period_fouls_before` | Home previous team fouls in period |
| `away_period_fouls_before` | Away previous team fouls in period |
| `period_foul_diff_home_minus_away_before` | Home period fouls minus away period fouls |
| `last_foul_against_home` | 1 if previous foul was against home |
| `time_since_last_foul` | Seconds since previous foul |
| `foul_type` | Parsed foul type |
| `shooting_foul` | 1 if shooting foul |
| `offensive_foul` | 1 if offensive foul |
| `loose_ball_foul` | 1 if loose-ball foul |
| `technical_foul` | 1 if technical foul |
| `flagrant_foul` | 1 if flagrant foul |
| `playoffs` | 1 if playoff game |
