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
from viewtrust.analysis.offline_multi_condition import (
    aggregate_condition_results,
    compute_condition_ranking,
    compute_failure_cases,
    condition_result_from_signal_dir,
    discover_offline_signal_dirs,
    validate_offline_signal_dir,
)
from viewtrust.analysis.statistics import NumericSummary, summarize_numbers, summarize_table
from viewtrust.analysis.tables import write_csv_table

__all__ = [
    "NumericSummary",
    "aggregate_condition_results",
    "compute_condition_ranking",
    "compute_failure_cases",
    "compute_ablation_metrics",
    "compute_group_metrics",
    "compute_signal_components",
    "condition_result_from_signal_dir",
    "discover_offline_signal_dirs",
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
    "validate_offline_signal_dir",
    "write_csv_table",
]
