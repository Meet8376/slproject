"""Automated Report Generator for the Kanban Engine.

Generates structured Markdown reports, CSV/Excel summaries, and terminal
dashboards using the NumPy & Pandas MetricsEngine.
"""

from datetime import datetime
from typing import Optional
import os
import pandas as pd

from .analytics import MetricsEngine
from .models import TaskStatus, Priority


class ReportGenerator:
    """Automates project status reporting in Markdown, CSV, and text."""

    def __init__(self, df: pd.DataFrame, project_name: str = "Software Lab Development Project"):
        self.df = df
        self.project_name = project_name
        self.metrics = MetricsEngine(df)

    def generate_terminal_summary(self) -> str:
        """Generate a concise textual dashboard for quick terminal reading."""
        summary = self.metrics.get_summary_metrics()
        stats = self.metrics.get_numpy_statistical_analysis()
        workload = self.metrics.get_assignee_workload()

        lines = [
            f"=== 📊 {self.project_name.upper()} - HEALTH SUMMARY ===",
            f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- High-Level Progress ---",
            f"Total Tasks        : {summary['total_tasks']}",
            f"Completed (DONE)   : {summary['completed_tasks']} ({summary['completion_rate_pct']}%)",
            f"In Progress        : {summary['in_progress_tasks']}",
            f"Backlog (TODO)     : {summary['todo_tasks']}",
            f"Blocked            : {summary['blocked_tasks']}",
            f"Total Hours Logged : {summary['total_actual_hours']} hrs (Estimated: {summary['total_estimated_hours']} hrs)",
            "",
            "--- NumPy Statistical Analysis (Practical 5) ---",
        ]

        if "error" not in stats:
            lines.extend([
                f"Mean Task Effort   : {stats['mean_actual_hours']} hrs (Std Dev: {stats['std_deviation']})",
                f"Median Effort (P50): {stats['percentile_50']} hrs",
                f"90th Percentile    : {stats['percentile_90']} hrs (90% of tasks completed within this effort)",
                f"Estimation Error   : {stats['mean_estimation_error']} hrs (Underestimated: {stats['underestimated_tasks_count']})",
                f"Outlier Tasks (> {stats['outlier_threshold']}h) : {', '.join(stats['outlier_tasks']) if stats['outlier_tasks'] else 'None'}",
            ])
        else:
            lines.append("No statistical data available.")

        lines.extend(["", "--- Team Workload ---"])
        if not workload.empty:
            for _, r in workload.iterrows():
                lines.append(f"• {r['assignee']:<12}: {r['done_tasks']}/{r['total_tasks']} tasks done ({r['completion_rate_pct']}%) | {r['actual_hours']} hrs logged")
        else:
            lines.append("No team data recorded.")

        return "\n".join(lines)

    @staticmethod
    def _df_to_markdown_table(df: pd.DataFrame) -> str:
        """Convert DataFrame to Markdown table with graceful fallback if tabulate is unavailable."""
        if df.empty:
            return "_No records found._"
        try:
            return df.to_markdown(index=False)
        except Exception:
            # Fallback markdown table generator
            headers = [str(c) for c in df.columns]
            header_row = "| " + " | ".join(headers) + " |"
            sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
            data_rows = []
            for _, r in df.iterrows():
                row_str = "| " + " | ".join(str(r[c]) for c in df.columns) + " |"
                data_rows.append(row_str)
            return "\n".join([header_row, sep_row] + data_rows)

    def generate_markdown_report(self, file_path: str = "project_report.md") -> str:
        """Generate and save an executive Markdown project report."""
        summary = self.metrics.get_summary_metrics()
        stats = self.metrics.get_numpy_statistical_analysis()
        workload = self.metrics.get_assignee_workload()
        categories = self.metrics.get_category_breakdown()

        md_content = f"""# 📋 {self.project_name} - Project Report

**Date of Report:** {datetime.now().strftime('%B %d, %Y - %H:%M')}  
**Core Technologies:** Python 3, Pandas (DataFrames & Vectorization), NumPy (Statistical Computations)  
**Academic Course:** Software Lab I (SL)

---

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Total Registered Tasks** | {summary['total_tasks']} |
| **Completed Tasks (DONE)** | {summary['completed_tasks']} |
| **Completion Rate** | **{summary['completion_rate_pct']}%** |
| **In Progress Tasks** | {summary['in_progress_tasks']} |
| **Backlog (TODO)** | {summary['todo_tasks']} |
| **Blocked Items** | {summary['blocked_tasks']} |
| **Total Estimated Effort** | {summary['total_estimated_hours']} hrs |
| **Total Actual Effort** | {summary['total_actual_hours']} hrs |

---

## 2. Statistical Analysis & Effort Distribution (NumPy Engine)

This section leverages **NumPy vectorization and statistical distributions** (as implemented in Practical 5):

* **Mean Task Effort:** `{stats.get('mean_actual_hours', 0)}` hours
* **Standard Deviation (sigma):** `{stats.get('std_deviation', 0)}` hours
* **Median Effort (50th Percentile):** `{stats.get('percentile_50', 0)}` hours
* **75th Percentile:** `{stats.get('percentile_75', 0)}` hours
* **90th Percentile:** `{stats.get('percentile_90', 0)}` hours
* **Estimation Variance:** Mean difference of `{stats.get('mean_estimation_error', 0)}` hours between actual and estimated effort.
* **Underestimated Tasks Count:** `{stats.get('underestimated_tasks_count', 0)}` tasks
* **Detected Effort Outliers:** `{', '.join(stats.get('outlier_tasks', [])) if stats.get('outlier_tasks') else 'None'}`

---

## 3. Team Member Workload Breakdown

{self._df_to_markdown_table(workload)}

## 4. Engineering Category Distribution

{self._df_to_markdown_table(categories)}
"""

        # Blocked / Critical Alert Section
        blocked_df = self.df[self.df["status"] == TaskStatus.BLOCKED]
        critical_df = self.df[self.df["priority"] == Priority.CRITICAL]

        md_content += "## 5. Risk & Escalation Register\n\n"
        if not blocked_df.empty:
            md_content += "### ⚠️ Blocked Tasks Requiring Attention\n"
            for _, r in blocked_df.iterrows():
                md_content += f"- **[{r['task_id']}] {r['title']}** (Assignee: @{r['assignee']})\n"
            md_content += "\n"
        else:
            md_content += "✅ *No blocked tasks currently on the board.*\n\n"

        if not critical_df.empty:
            md_content += "### 🚨 Critical Priority Tasks\n"
            for _, r in critical_df.iterrows():
                md_content += f"- **[{r['task_id']}] {r['title']}** - Status: `{r['status']}` (Assignee: @{r['assignee']})\n"
            md_content += "\n"

        # Machine Learning Predictive Insights Section
        try:
            from .ml import KanbanMLAssistant
            ml_assistant = KanbanMLAssistant()
            ml_assistant.train(self.df)
            reg_m = ml_assistant.regression_metrics
            risk_m = ml_assistant.risk_metrics
            weights = ml_assistant.get_feature_importances()

            active_tasks = self.df[self.df["status"].isin([TaskStatus.TODO, TaskStatus.IN_PROGRESS, TaskStatus.REVIEW])]
            active_predictions = []
            for _, task_row in active_tasks.iterrows():
                p_effort = ml_assistant.predict_task_effort(
                    estimated_hours=float(task_row.get("estimated_hours", 2.0)),
                    priority=str(task_row.get("priority", "MEDIUM")),
                    category=str(task_row.get("category", "General")),
                    assignee=str(task_row.get("assignee", "Meet")),
                    tags=str(task_row.get("tags", "")),
                    title=str(task_row.get("title", "")),
                )
                p_risk = ml_assistant.predict_overrun_risk(
                    estimated_hours=float(task_row.get("estimated_hours", 2.0)),
                    priority=str(task_row.get("priority", "MEDIUM")),
                    category=str(task_row.get("category", "General")),
                    assignee=str(task_row.get("assignee", "Meet")),
                    tags=str(task_row.get("tags", "")),
                    title=str(task_row.get("title", "")),
                )
                active_predictions.append({
                    "task_id": task_row["task_id"],
                    "title": task_row["title"][:28] + "..." if len(str(task_row["title"])) > 28 else task_row["title"],
                    "assignee": task_row["assignee"],
                    "status": task_row["status"],
                    "estimated_hrs": task_row["estimated_hours"],
                    "predicted_hrs": p_effort["predicted_actual_hours"],
                    "variance": f"{p_effort['predicted_variance']:+0.1f}h",
                    "overrun_risk": f"{p_risk['risk_symbol']} {p_risk['risk_level']} ({p_risk['risk_probability_pct']}%)",
                })

            pred_df = pd.DataFrame(active_predictions)

            md_content += "## 6. 🤖 Machine Learning Predictive Insights (NumPy Engine)\n\n"
            md_content += "Trained on historical task outcomes via **Multiple Linear Regression** and **Logistic Regression** (Gradient Descent):\n\n"
            md_content += f"* **Effort Predictor R² Score:** `{reg_m.get('r2_score', 0.0)}` (Mean Absolute Error: `{reg_m.get('mae', 0.0)}` hrs)\n"
            md_content += f"* **Risk Classifier Accuracy:** `{risk_m.get('accuracy', 0.0)}%` (Precision: `{risk_m.get('precision', 0.0)}%`)\n\n"
            
            md_content += "### Key Influencing Features (Learned Weights):\n"
            for feat_name, w in weights[:4]:
                direction = "increases effort" if w > 0 else "reduces effort"
                md_content += f"- **`{feat_name}`**: Weight `{w:+.3f}` ({direction})\n"
            md_content += "\n"

            md_content += "### Active Task Predictive Audit:\n"
            md_content += self._df_to_markdown_table(pred_df) + "\n\n"
        except Exception as e:
            md_content += f"<!-- ML Section skipped: {e} -->\n\n"

        md_content += "---\n*Generated automatically by Python Terminal Kanban Engine.*"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            return file_path
        except Exception as e:
            print(f"Error saving report: {e}")
            return ""

