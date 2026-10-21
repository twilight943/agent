from collections.abc import Sequence

import pandas as pd


def user_history_tool(
    interactions: pd.DataFrame, user_id: int, limit: int = 50
) -> list[dict[str, object]]:
    rows = interactions[interactions["user_id"] == user_id].sort_values("timestamp").tail(limit)
    return rows.to_dict(orient="records")
