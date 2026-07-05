"""Small statistics helpers for Priority 0 observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


@dataclass(frozen=True)
class NumericSummary:
    """Summary statistics for one numeric series."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    population_std: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "population_std": self.population_std,
        }


def summarize_numbers(values: Iterable[float | int | None]) -> NumericSummary:
    """Return deterministic summary stats for numeric values."""

    numbers = [float(value) for value in values if value is not None]
    if not numbers:
        return NumericSummary(
            count=0,
            minimum=None,
            maximum=None,
            mean=None,
            population_std=None,
        )

    count = len(numbers)
    mean = sum(numbers) / count
    variance = sum((value - mean) ** 2 for value in numbers) / count
    return NumericSummary(
        count=count,
        minimum=min(numbers),
        maximum=max(numbers),
        mean=mean,
        population_std=sqrt(variance),
    )


def summarize_table(
    rows: Iterable[dict[str, object]],
    numeric_fields: Iterable[str],
) -> dict[str, dict[str, float | int | None]]:
    """Summarize selected numeric fields from row dictionaries."""

    row_list = list(rows)
    return {
        field: summarize_numbers(
            value if isinstance(value, (int, float)) else None
            for value in (row.get(field) for row in row_list)
        ).as_dict()
        for field in numeric_fields
    }
