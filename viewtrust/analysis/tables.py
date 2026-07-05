"""Table helpers for Priority 0 CSV artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_csv_table(
    path: Path,
    rows: Iterable[dict[str, object]],
    fieldnames: Iterable[str],
) -> Path:
    """Write a stable CSV table with explicit column order."""

    row_list = list(rows)
    ordered_fields = list(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields, extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            writer.writerow({field: row.get(field, "") for field in ordered_fields})

    return path
