"""Terminal Visualizer for the Kanban Board.

Renders interactive, formatted ASCII columns and task summary tables
optimized for terminal-driven developer workflows.
"""

import shutil
from typing import Dict, List, Optional
import pandas as pd

from .models import Task, TaskStatus, Priority


class Colors:
    """ANSI color codes for terminal styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Status colors
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    BLUE = "\033[34m"

    # Priority tags
    PRIORITY_COLORS = {
        Priority.CRITICAL: RED,
        Priority.HIGH: YELLOW,
        Priority.MEDIUM: CYAN,
        Priority.LOW: BLUE,
    }


class BoardVisualizer:
    """Renders Kanban columns and tables in the terminal."""

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate string with ellipsis if exceeds max_len."""
        text = str(text)
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."

    @classmethod
    def render_board(cls, columns: Dict[str, List[Task]], use_color: bool = True) -> str:
        """Render side-by-side or stacked Kanban columns based on terminal width."""
        term_width = shutil.get_terminal_size((120, 30)).columns
        active_statuses = [TaskStatus.TODO, TaskStatus.IN_PROGRESS, TaskStatus.REVIEW, TaskStatus.DONE]
        
        # If there are blocked tasks, include BLOCKED column
        if columns.get(TaskStatus.BLOCKED):
            active_statuses.append(TaskStatus.BLOCKED)

        col_count = len(active_statuses)
        col_width = max(24, (term_width - (col_count * 3)) // col_count)

        lines: List[str] = []
        lines.append("\n" + "=" * term_width)
        title = " 📋 TERMINAL KANBAN BOARD (Pandas & NumPy Engine) "
        lines.append(f"{title.center(term_width, '=')}")
        lines.append("=" * term_width + "\n")

        # Column Headers
        header_parts = []
        for status in active_statuses:
            task_list = columns.get(status, [])
            header_text = f" {status} ({len(task_list)}) "
            header_parts.append(header_text.center(col_width, "─"))
        lines.append(" | ".join(header_parts))
        lines.append("─" * term_width)

        # Build card representations for each column
        max_cards = max((len(columns.get(s, [])) for s in active_statuses), default=0)

        if max_cards == 0:
            lines.append(" Board is currently empty. Use 'add' to create new tasks.\n".center(term_width))
            return "\n".join(lines)

        for card_idx in range(max_cards):
            # Row 1: Task ID & Priority
            row1_parts = []
            for status in active_statuses:
                tasks = columns.get(status, [])
                if card_idx < len(tasks):
                    t = tasks[card_idx]
                    p_color = Colors.PRIORITY_COLORS.get(t.priority, "") if use_color else ""
                    reset = Colors.RESET if use_color else ""
                    tid_str = f"[{t.task_id}]"
                    p_str = f"{p_color}{t.priority[:4]}{reset}"
                    # Total visible length calculation
                    visible_len = len(tid_str) + 1 + len(t.priority[:4])
                    padding = max(0, col_width - visible_len)
                    cell = f"{tid_str} {p_str}" + (" " * padding)
                    row1_parts.append(cell)
                else:
                    row1_parts.append(" " * col_width)
            lines.append(" │ ".join(row1_parts))

            # Row 2: Title
            row2_parts = []
            for status in active_statuses:
                tasks = columns.get(status, [])
                if card_idx < len(tasks):
                    t = tasks[card_idx]
                    short_title = cls._truncate(t.title, col_width - 2)
                    row2_parts.append(f" {short_title:<{col_width-1}}")
                else:
                    row2_parts.append(" " * col_width)
            lines.append(" │ ".join(row2_parts))

            # Row 3: Assignee & Hours
            row3_parts = []
            for status in active_statuses:
                tasks = columns.get(status, [])
                if card_idx < len(tasks):
                    t = tasks[card_idx]
                    info = f"@{cls._truncate(t.assignee, 10)} | {t.actual_hours}h/{t.estimated_hours}h"
                    short_info = cls._truncate(info, col_width - 2)
                    row3_parts.append(f" {short_info:<{col_width-1}}")
                else:
                    row3_parts.append(" " * col_width)
            lines.append(" │ ".join(row3_parts))

            # Divider between cards
            lines.append("─" * term_width)

        return "\n".join(lines)

    @classmethod
    def render_task_table(cls, df: pd.DataFrame) -> str:
        """Render a clean ASCII table of a filtered DataFrame."""
        if df.empty:
            return "No tasks found matching the criteria."

        display_cols = ["task_id", "title", "assignee", "status", "priority", "category", "actual_hours", "estimated_hours"]
        available_cols = [c for c in display_cols if c in df.columns]
        
        # Calculate max column widths
        widths = {}
        for col in available_cols:
            max_val_len = df[col].astype(str).str.len().max()
            widths[col] = max(len(col), max_val_len, 6)

        # Header
        header = " | ".join(f"{col.upper():<{widths[col]}}" for col in available_cols)
        divider = "-+-".join("-" * widths[col] for col in available_cols)

        rows = []
        rows.append(header)
        rows.append(divider)

        for _, r in df.iterrows():
            row_str = " | ".join(f"{str(r[col]):<{widths[col]}}" for col in available_cols)
            rows.append(row_str)

        return "\n".join(rows)
