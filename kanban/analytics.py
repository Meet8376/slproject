"""Analytics and Metrics Engine.

Leverages NumPy and Pandas to compute project metrics, statistical summaries,
percentile analyses, estimation variances, and workload distributions.
Directly implements concepts from Practical 5 (NumPy arrays, stats, percentiles)
and Pandas groupbys/aggregations.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .models import TaskStatus, Priority


class MetricsEngine:
    """Calculates project metrics using NumPy statistical functions and Pandas aggregations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Compute top-level summary metrics of the Kanban board."""
        total_tasks = len(self.df)
        if total_tasks == 0:
            return {
                "total_tasks": 0,
                "completed_tasks": 0,
                "in_progress_tasks": 0,
                "todo_tasks": 0,
                "blocked_tasks": 0,
                "completion_rate_pct": 0.0,
                "total_estimated_hours": 0.0,
                "total_actual_hours": 0.0,
            }

        status_counts = self.df.groupby("status").size().to_dict()
        done_count = status_counts.get(TaskStatus.DONE, 0)
        in_progress_count = status_counts.get(TaskStatus.IN_PROGRESS, 0)
        todo_count = status_counts.get(TaskStatus.TODO, 0)
        blocked_count = status_counts.get(TaskStatus.BLOCKED, 0)

        completion_rate = (done_count / total_tasks) * 100.0 if total_tasks > 0 else 0.0

        total_est = float(self.df["estimated_hours"].sum())
        total_act = float(self.df["actual_hours"].sum())

        return {
            "total_tasks": total_tasks,
            "completed_tasks": done_count,
            "in_progress_tasks": in_progress_count,
            "todo_tasks": todo_count,
            "blocked_tasks": blocked_count,
            "completion_rate_pct": round(completion_rate, 2),
            "total_estimated_hours": round(total_est, 2),
            "total_actual_hours": round(total_act, 2),
        }

    def get_numpy_statistical_analysis(self) -> Dict[str, Any]:
        """Compute statistical measures on task hours using NumPy (Practical 5)."""
        if self.df.empty:
            return {"error": "No tasks available for statistical analysis."}

        # Convert Pandas Series to NumPy arrays
        actual_arr = np.array(self.df["actual_hours"].values, dtype=float)
        est_arr = np.array(self.df["estimated_hours"].values, dtype=float)

        # Statistical calculations (NumPy functions taught in Practical 5)
        mean_actual = float(np.mean(actual_arr))
        median_actual = float(np.median(actual_arr))
        var_actual = float(np.var(actual_arr))
        std_actual = float(np.std(actual_arr))
        max_actual = float(np.max(actual_arr))
        min_actual = float(np.min(actual_arr))

        # Percentile calculations (np.percentile taught in Practical 5 Q3)
        p50 = float(np.percentile(actual_arr, 50))
        p75 = float(np.percentile(actual_arr, 75))
        p90 = float(np.percentile(actual_arr, 90))

        # Estimation variance analysis (Vectorized NumPy subtraction)
        variance_arr = actual_arr - est_arr
        mean_estimation_error = float(np.mean(variance_arr))
        
        # Boolean masking (NumPy boolean indexing taught in Practical 5 Q9)
        underestimated_count = int(np.sum(variance_arr > 0))  # took longer than planned
        on_track_count = int(np.sum(variance_arr == 0))
        overestimated_count = int(np.sum(variance_arr < 0))

        # Outlier detection: tasks where actual hours exceed 75th percentile + 1.5 * std
        threshold = p75 + (1.5 * std_actual) if std_actual > 0 else p75
        outlier_mask = actual_arr >= threshold
        outlier_task_ids = list(self.df.loc[outlier_mask, "task_id"].values)

        return {
            "mean_actual_hours": round(mean_actual, 2),
            "median_actual_hours": round(median_actual, 2),
            "variance": round(var_actual, 2),
            "std_deviation": round(std_actual, 2),
            "min_hours": round(min_actual, 2),
            "max_hours": round(max_actual, 2),
            "percentile_50": round(p50, 2),
            "percentile_75": round(p75, 2),
            "percentile_90": round(p90, 2),
            "mean_estimation_error": round(mean_estimation_error, 2),
            "underestimated_tasks_count": underestimated_count,
            "on_track_tasks_count": on_track_count,
            "overestimated_tasks_count": overestimated_count,
            "outlier_threshold": round(threshold, 2),
            "outlier_tasks": outlier_task_ids,
        }

    def get_assignee_workload(self) -> pd.DataFrame:
        """Compute workload distribution per team member using Pandas groupby."""
        if self.df.empty:
            return pd.DataFrame(columns=["assignee", "total_tasks", "done_tasks", "actual_hours", "completion_rate"])

        # Group by assignee and aggregate
        grouped = self.df.groupby("assignee")
        workload = pd.DataFrame({
            "total_tasks": grouped.size(),
            "done_tasks": grouped.apply(lambda g: (g["status"] == TaskStatus.DONE).sum()),
            "in_progress": grouped.apply(lambda g: (g["status"] == TaskStatus.IN_PROGRESS).sum()),
            "estimated_hours": grouped["estimated_hours"].sum(),
            "actual_hours": grouped["actual_hours"].sum(),
        }).reset_index()

        workload["completion_rate_pct"] = (
            (workload["done_tasks"] / workload["total_tasks"]) * 100
        ).round(1)

        return workload.sort_values(by="total_tasks", ascending=False)

    def get_category_breakdown(self) -> pd.DataFrame:
        """Analyze tasks by engineering category."""
        if self.df.empty:
            return pd.DataFrame()

        cat_summary = self.df.groupby("category").agg(
            task_count=("task_id", "count"),
            total_hours=("actual_hours", "sum"),
            avg_hours=("actual_hours", "mean"),
        ).reset_index()

        cat_summary["avg_hours"] = cat_summary["avg_hours"].round(1)
        return cat_summary.sort_values(by="task_count", ascending=False)

    def get_priority_matrix(self) -> pd.DataFrame:
        """Generate a cross-tabulation of Priority vs Status."""
        if self.df.empty:
            return pd.DataFrame()

        # Pandas crosstabulation
        return pd.crosstab(self.df["priority"], self.df["status"], margins=True)
