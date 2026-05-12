# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DataProcessor:
    def prepare_optimizer_input(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            results.append(
                {
                    "university": row.get("university", ""),
                    "major": row.get("major", ""),
                    "probability": float(row.get("probability", 0.0)),
                    "admitted": int(row.get("admitted", 0)),
                    "similarity": float(row.get("similarity", 0.0)),
                }
            )
        return results
