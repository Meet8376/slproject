"""Core Kanban Engine using Pandas DataFrames.

Implements fast indexing, dynamic state updates, state transition validation,
and bulk task filtering using Pandas.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

from .models import Task, TaskStatus, Priority, validate_transition
from .storage import StorageEngine, SCHEMA_COLUMNS


class KanbanEngine:
    """Core Kanban data processing engine powered by Pandas."""

    def __init__(self, storage_path: str = "tasks.csv"):
        self.storage = StorageEngine(storage_path)
        self.df = self.storage.load()
        self._sync_index()

    def _sync_index(self) -> None:
        """Ensure task_id is cleaned and set as index for fast O(1) lookups."""
        if not self.df.empty:
            self.df["task_id"] = self.df["task_id"].astype(str).str.strip().str.upper()
            # Drop duplicates if any, keeping first
            self.df = self.df.drop_duplicates(subset=["task_id"], keep="first")
            self.df.set_index("task_id", drop=False, inplace=True)
            self.df.index.name = "task_index"

    def refresh(self) -> None:
        """Reload DataFrame from persistent storage."""
        self.df = self.storage.load()
        self._sync_index()

    def persist(self) -> bool:
        """Persist current DataFrame to storage."""
        return self.storage.save(self.df)

    def generate_next_task_id(self, prefix: str = "TASK-") -> str:
        """Auto-generate the next sequential task ID (e.g., TASK-101)."""
        if self.df.empty:
            return f"{prefix}101"

        numeric_ids = []
        for tid in self.df["task_id"]:
            tid_str = str(tid).strip()
            if tid_str.startswith(prefix):
                num_part = tid_str[len(prefix):]
                if num_part.isdigit():
                    numeric_ids.append(int(num_part))

        if numeric_ids:
            return f"{prefix}{max(numeric_ids) + 1}"
        return f"{prefix}101"

    # ==========================
    # CRUD Operations
    # ==========================

    def add_task(self, task: Task) -> Tuple[bool, str]:
        """Add a new task to the Kanban board."""
        tid = task.task_id
        if not tid:
            tid = self.generate_next_task_id()
            task.task_id = tid

        if not self.df.empty and tid in self.df.index:
            return False, f"Task ID '{tid}' already exists on the board."

        # Convert to dictionary and create a 1-row DataFrame
        new_row = pd.DataFrame([task.to_dict()])
        new_row.set_index("task_id", drop=False, inplace=True)
        new_row.index.name = "task_index"

        if self.df.empty:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row])

        self.persist()
        return True, f"Task '{tid}' ('{task.title}') added successfully."

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID using Pandas fast .loc indexing."""
        tid = str(task_id).strip().upper()
        if self.df.empty or tid not in self.df.index:
            return None
        row = self.df.loc[tid].to_dict()
        return Task.from_dict(row)

    def update_task(self, task_id: str, **kwargs) -> Tuple[bool, str]:
        """Dynamically update task fields using Pandas .loc indexing."""
        tid = str(task_id).strip().upper()
        if self.df.empty or tid not in self.df.index:
            return False, f"Task '{tid}' not found."

        updated_fields = []
        for key, value in kwargs.items():
            if key in SCHEMA_COLUMNS and key != "task_id":
                if key in ["estimated_hours", "actual_hours"]:
                    self.df.loc[tid, key] = float(value)
                elif key == "tags":
                    self.df.loc[tid, key] = Task._clean_tags(value)
                elif key == "status":
                    self.df.loc[tid, key] = str(value).upper()
                elif key == "priority":
                    self.df.loc[tid, key] = str(value).upper()
                else:
                    self.df.loc[tid, key] = str(value)
                updated_fields.append(key)

        if updated_fields:
            self.persist()
            return True, f"Task '{tid}' updated: {', '.join(updated_fields)}"
        return False, "No valid fields were provided for update."

    def delete_task(self, task_id: str) -> Tuple[bool, str]:
        """Remove a task from the board by task_id."""
        tid = str(task_id).strip().upper()
        if self.df.empty or tid not in self.df.index:
            return False, f"Task '{tid}' not found."

        self.df = self.df.drop(index=tid)
        self.persist()
        return True, f"Task '{tid}' successfully deleted."

    # ==========================
    # State-Transition Logic
    # ==========================

    def move_task(self, task_id: str, new_status: str) -> Tuple[bool, str]:
        """Move task to a new status with validation against allowed state transitions."""
        tid = str(task_id).strip().upper()
        target_status = str(new_status).strip().upper()

        task = self.get_task(tid)
        if not task:
            return False, f"Task '{tid}' not found."

        current_status = task.status
        is_valid, reason = validate_transition(current_status, target_status)

        if not is_valid:
            return False, f"Cannot move {tid}: {reason}"

        # Apply state transition
        self.df.loc[tid, "status"] = target_status

        # If completed, stamp completed_at timestamp; if reopened, clear timestamp
        if target_status == TaskStatus.DONE:
            self.df.loc[tid, "completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        elif current_status == TaskStatus.DONE and target_status != TaskStatus.DONE:
            self.df.loc[tid, "completed_at"] = ""

        self.persist()
        return True, f"Task {tid} moved: {current_status} -> {target_status}."

    # ==========================
    # Bulk Task Filtering & Slicing
    # ==========================

    def filter_tasks(
        self,
        status: Optional[str] = None,
        assignee: Optional[str] = None,
        priority: Optional[str] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        min_hours: Optional[float] = None,
        max_hours: Optional[float] = None,
    ) -> pd.DataFrame:
        """Perform bulk filtering using Pandas vectorized boolean indexing."""
        if self.df.empty:
            return self.df.copy()

        filtered = self.df.copy()

        if status:
            s_val = status.strip().upper()
            filtered = filtered[filtered["status"] == s_val]

        if assignee:
            a_val = assignee.strip().lower()
            filtered = filtered[filtered["assignee"].str.lower() == a_val]

        if priority:
            p_val = priority.strip().upper()
            filtered = filtered[filtered["priority"] == p_val]

        if category:
            c_val = category.strip().lower()
            filtered = filtered[filtered["category"].str.lower() == c_val]

        if tag:
            t_val = tag.strip().lower()
            filtered = filtered[filtered["tags"].str.lower().str.contains(t_val, na=False)]

        if min_hours is not None:
            filtered = filtered[filtered["actual_hours"] >= min_hours]

        if max_hours is not None:
            filtered = filtered[filtered["actual_hours"] <= max_hours]

        return filtered

    def get_board_columns(self) -> Dict[str, List[Task]]:
        """Return tasks grouped by column for board visualization."""
        columns = {s: [] for s in TaskStatus.ALL_STATUSES}
        if self.df.empty:
            return columns

        for _, row in self.df.iterrows():
            task = Task.from_dict(row.to_dict())
            if task.status in columns:
                columns[task.status].append(task)
            else:
                columns.setdefault(task.status, []).append(task)
        return columns

    def sort_tasks(self, by: str = "priority", ascending: bool = False) -> pd.DataFrame:
        """Sort tasks using Pandas sort_values with custom weights."""
        if self.df.empty:
            return self.df.copy()

        df_copy = self.df.copy()
        if by == "priority":
            # Map priority string to numeric weight
            weights = Priority.PRIORITY_WEIGHTS
            df_copy["_p_weight"] = df_copy["priority"].map(lambda p: weights.get(p, 0))
            df_sorted = df_copy.sort_values(by="_p_weight", ascending=ascending)
            return df_sorted.drop(columns=["_p_weight"])
        elif by in df_copy.columns:
            return df_copy.sort_values(by=by, ascending=ascending)
        return df_copy
