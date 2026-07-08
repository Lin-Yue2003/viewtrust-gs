"""Analysis helpers for future Priority 0 observations."""

from viewtrust.analysis.offline_signals import (
    compute_ablation_metrics,
    compute_group_metrics,
    compute_signal_components,
    mad,
    positive_part,
    precision_at_k,
    rank_descending,
    recall_at_k,
    robust_median,
    robust_z_scores,
    safe_divide,
    safe_float,
)
from viewtrust.analysis.statistics import NumericSummary, summarize_numbers, summarize_table
from viewtrust.analysis.tables import write_csv_table

__all__ = [
    "NumericSummary",
    "compute_ablation_metrics",
    "compute_group_metrics",
    "compute_signal_components",
    "mad",
    "positive_part",
    "precision_at_k",
    "rank_descending",
    "recall_at_k",
    "robust_median",
    "robust_z_scores",
    "safe_divide",
    "safe_float",
    "summarize_numbers",
    "summarize_table",
    "write_csv_table",
]
