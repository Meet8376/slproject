"""Storage layer for the Kanban Engine.

Uses Pandas DataFrames to load and persist task state directly to/from
CSV or Excel (.xlsx) files.
"""

import os
import shutil
from datetime import datetime
from typing import Optional
import pandas as pd


SCHEMA_COLUMNS = [
    "task_id",
    "title",
    "assignee",
    "status",
    "priority",
    "category",
    "estimated_hours",
    "actual_hours",
    "tags",
    "created_at",
    "completed_at",
]


class StorageEngine:
    """Manages file persistence (CSV / Excel) using Pandas DataFrames."""

    def __init__(self, file_path: str = "tasks.csv"):
        self.file_path = file_path
        self._ensure_storage_exists()

    @property
    def is_excel(self) -> bool:
        """Check if the target storage format is Excel."""
        return self.file_path.lower().endswith((".xlsx", ".xls"))

    def _ensure_storage_exists(self) -> None:
        """Create initial empty storage file with defined schema if it does not exist."""
        if not os.path.exists(self.file_path):
            empty_df = pd.DataFrame(columns=SCHEMA_COLUMNS)
            self.save(empty_df)

    def load(self) -> pd.DataFrame:
        """Load tasks DataFrame from CSV or Excel file."""
        if not os.path.exists(self.file_path):
            return pd.DataFrame(columns=SCHEMA_COLUMNS)

        try:
            if self.is_excel:
                df = pd.read_excel(self.file_path, engine="openpyxl")
            else:
                df = pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Warning: Failed to load {self.file_path} ({e}). Creating blank board.")
            df = pd.DataFrame(columns=SCHEMA_COLUMNS)

        # Ensure all standard schema columns are present
        for col in SCHEMA_COLUMNS:
            if col not in df.columns:
                df[col] = "" if "hours" not in col else 0.0

        # Fill missing values and ensure type consistency
        df["task_id"] = df["task_id"].fillna("").astype(str)
        df["title"] = df["title"].fillna("").astype(str)
        df["assignee"] = df["assignee"].fillna("Unassigned").astype(str)
        df["status"] = df["status"].fillna("TODO").astype(str).str.upper()
        df["priority"] = df["priority"].fillna("MEDIUM").astype(str).str.upper()
        df["category"] = df["category"].fillna("General").astype(str)
        df["estimated_hours"] = pd.to_numeric(df["estimated_hours"], errors="coerce").fillna(0.0)
        df["actual_hours"] = pd.to_numeric(df["actual_hours"], errors="coerce").fillna(0.0)
        df["tags"] = df["tags"].fillna("").astype(str)
        df["created_at"] = df["created_at"].fillna("").astype(str)
        df["completed_at"] = df["completed_at"].fillna("").astype(str)

        return df[SCHEMA_COLUMNS]

    def save(self, df: pd.DataFrame) -> bool:
        """Persist DataFrame to CSV or Excel file."""
        try:
            # Reorder columns according to standard schema
            export_df = df.copy()
            for col in SCHEMA_COLUMNS:
                if col not in export_df.columns:
                    export_df[col] = "" if "hours" not in col else 0.0
            export_df = export_df[SCHEMA_COLUMNS]

            if self.is_excel:
                export_df.to_excel(self.file_path, index=False, engine="openpyxl")
            else:
                export_df.to_csv(self.file_path, index=False)
            return True
        except Exception as e:
            print(f"Error persisting tasks to {self.file_path}: {e}")
            return False

    def create_backup(self, backup_dir: str = "backups") -> Optional[str]:
        """Create a timestamped backup copy of current storage."""
        if not os.path.exists(self.file_path):
            return None
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = ".xlsx" if self.is_excel else ".csv"
        backup_filename = os.path.join(backup_dir, f"tasks_backup_{timestamp}{ext}")
        shutil.copyfile(self.file_path, backup_filename)
        return backup_filename

    def switch_format(self, new_file_path: str) -> pd.DataFrame:
        """Switch storage to another file or format (e.g., CSV to Excel)."""
        df = self.load()
        self.file_path = new_file_path
        self.save(df)
        return df
