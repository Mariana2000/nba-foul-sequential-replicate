import pandas as pd
from pathlib import Path

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_game_id(game_id: str | int) -> str:
    """Normalize NBA game IDs to 10-digit zero-padded strings."""
    return str(game_id).zfill(10)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    out = df.copy()
    if "game_id" in out.columns:
        out["game_id"] = out["game_id"].map(normalize_game_id)
    out.to_csv(path, index=False)
