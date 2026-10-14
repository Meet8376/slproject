"""Task data model and state-transition definitions.

Implements concepts taught in Software Lab:
- Object-Oriented Programming (Classes, __init__, methods - Practical 4)
- Dictionaries, Sets, and Tuples (Practical 2 & 3)
"""

from datetime import datetime
from typing import Dict, Set, Optional, Any, Tuple


class TaskStatus:
    """Allowed states for tasks on the Kanban board."""
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    DONE = "DONE"
    BLOCKED = "BLOCKED"

    ALL_STATUSES = [TODO, IN_PROGRESS, REVIEW, DONE, BLOCKED]


class Priority:
    """Priority levels for tasks."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    ALL_PRIORITIES = [LOW, MEDIUM, HIGH, CRITICAL]
    PRIORITY_WEIGHTS = {LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4}


# State transition rules: maps current state to a set of permitted next states
# Practical 3 concept: Set lookups and dictionary mapping
VALID_TRANSITIONS: Dict[str, Set[str]] = {
    TaskStatus.TODO: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED},
    TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.BLOCKED, TaskStatus.TODO},
    TaskStatus.REVIEW: {TaskStatus.DONE, TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED},
    TaskStatus.BLOCKED: {TaskStatus.TODO, TaskStatus.IN_PROGRESS},
    TaskStatus.DONE: {TaskStatus.TODO},  # Allowed to reopen a completed task
}


def validate_transition(current_status: str, new_status: str) -> Tuple[bool, str]:
    """Validate whether transitioning from current_status to new_status is permitted.
    
    Returns:
        (is_valid, reason_message)
    """
    current = current_status.strip().upper()
    target = new_status.strip().upper()

    if target not in TaskStatus.ALL_STATUSES:
        return False, f"Invalid target status '{target}'. Must be one of: {TaskStatus.ALL_STATUSES}"

    if current == target:
        return False, f"Task is already in status '{target}'."

    allowed = VALID_TRANSITIONS.get(current, set())
    if target in allowed:
        return True, f"Transition from {current} to {target} is allowed."

    allowed_list = sorted(list(allowed))
    return False, f"Illegal transition from '{current}' to '{target}'. Allowed target states: {allowed_list}"


class Task:
    """Represents a single task in the Kanban system (Practical 4 OOP model)."""

    def __init__(
        self,
        task_id: str,
        title: str,
        assignee: str,
        status: str = TaskStatus.TODO,
        priority: str = Priority.MEDIUM,
        category: str = "General",
        estimated_hours: float = 0.0,
        actual_hours: float = 0.0,
        tags: str = "",
        created_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        self.task_id = str(task_id).strip().upper()
        self.title = str(title).strip()
        self.assignee = str(assignee).strip()
        self.status = str(status).strip().upper()
        self.priority = str(priority).strip().upper()
        self.category = str(category).strip()
        self.estimated_hours = float(estimated_hours)
        self.actual_hours = float(actual_hours)
        self.tags = self._clean_tags(tags)
        self.created_at = created_at or datetime.now().strftime("%Y-%m-%d %H:%M")
        self.completed_at = completed_at if completed_at else ""

    @staticmethod
    def _clean_tags(tags_input: Any) -> str:
        """Parse, deduplicate and sort tags using Python sets (Practical 2 & 3)."""
        if not tags_input or str(tags_input).strip() == "" or str(tags_input).lower() == "nan":
            return ""
        if isinstance(tags_input, (list, set, tuple)):
            clean_set = {str(t).strip().lower() for t in tags_input if str(t).strip()}
        else:
            clean_set = {t.strip().lower() for t in str(tags_input).split(",") if t.strip()}
        return ", ".join(sorted(clean_set))

    def get_tag_set(self) -> Set[str]:
        """Return tags as a Python set for set operations (intersection/union)."""
        if not self.tags:
            return set()
        return {t.strip().lower() for t in self.tags.split(",") if t.strip()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert task object to dictionary for Pandas DataFrame consumption."""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "assignee": self.assignee,
            "status": self.status,
            "priority": self.priority,
            "category": self.category,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "tags": self.tags,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Instantiate a Task from a dictionary row."""
        return cls(
            task_id=data.get("task_id", ""),
            title=data.get("title", ""),
            assignee=data.get("assignee", "Unassigned"),
            status=data.get("status", TaskStatus.TODO),
            priority=data.get("priority", Priority.MEDIUM),
            category=data.get("category", "General"),
            estimated_hours=float(data.get("estimated_hours", 0.0) or 0.0),
            actual_hours=float(data.get("actual_hours", 0.0) or 0.0),
            tags=str(data.get("tags", "") or ""),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at", ""),
        )

    def __repr__(self) -> str:
        return f"<Task {self.task_id} [{self.status}] '{self.title}' ({self.assignee})>"
