"""Small reporting helpers for pilot and formal experiment tables."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dirs(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, float_digits: int = 6) -> str:
    """Render a DataFrame as a dependency-free Markdown table."""
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.{float_digits}g}")
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = [str(col) for col in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.values.tolist():
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
