"""Sample dataset generator for the Kanban Engine.

Populates initial realistic tasks across the team members to showcase
all board columns, state transitions, filtering, and NumPy metrics.
"""

from kanban.engine import KanbanEngine
from kanban.models import Task, TaskStatus, Priority


def populate_sample_data(engine: KanbanEngine) -> None:
    """Populate Kanban board with representative sample tasks."""
    tasks = [
        Task(
            task_id="TASK-101",
            title="Design Pandas schema & file persistence layer",
            assignee="Meet",
            status=TaskStatus.DONE,
            priority=Priority.HIGH,
            category="Backend",
            estimated_hours=6.0,
            actual_hours=5.5,
            tags="pandas, schema, storage",
            created_at="2026-10-10 10:00",
            completed_at="2026-10-11 16:30",
        ),
        Task(
            task_id="TASK-102",
            title="Implement NumPy statistical metrics engine",
            assignee="Sujal",
            status=TaskStatus.DONE,
            priority=Priority.HIGH,
            category="Analytics",
            estimated_hours=5.0,
            actual_hours=6.0,
            tags="numpy, statistics, math",
            created_at="2026-10-10 11:30",
            completed_at="2026-10-12 14:00",
        ),
        Task(
            task_id="TASK-103",
            title="Create multi-column ASCII terminal visualizer",
            assignee="Member3",
            status=TaskStatus.DONE,
            priority=Priority.MEDIUM,
            category="CLI",
            estimated_hours=4.0,
            actual_hours=3.5,
            tags="ui, terminal, visualizer",
            created_at="2026-10-11 09:00",
            completed_at="2026-10-12 18:00",
        ),
        Task(
            task_id="TASK-104",
            title="Add state transition validation & constraint rules",
            assignee="Meet",
            status=TaskStatus.REVIEW,
            priority=Priority.CRITICAL,
            category="Core",
            estimated_hours=4.5,
            actual_hours=5.0,
            tags="validation, state-machine, core",
            created_at="2026-10-12 10:00",
        ),
        Task(
            task_id="TASK-105",
            title="Implement bulk DataFrame boolean filtering engine",
            assignee="Sujal",
            status=TaskStatus.IN_PROGRESS,
            priority=Priority.HIGH,
            category="Backend",
            estimated_hours=5.0,
            actual_hours=4.0,
            tags="pandas, filtering, search",
            created_at="2026-10-13 09:30",
        ),
        Task(
            task_id="TASK-106",
            title="Build automated Markdown & CSV report generator",
            assignee="Member3",
            status=TaskStatus.IN_PROGRESS,
            priority=Priority.MEDIUM,
            category="Reporting",
            estimated_hours=3.5,
            actual_hours=2.5,
            tags="reports, markdown, export",
            created_at="2026-10-13 11:00",
        ),
        Task(
            task_id="TASK-107",
            title="Support Excel (.xlsx) storage format via openpyxl",
            assignee="Meet",
            status=TaskStatus.TODO,
            priority=Priority.MEDIUM,
            category="Storage",
            estimated_hours=3.0,
            actual_hours=0.0,
            tags="excel, openpyxl, persistence",
            created_at="2026-10-14 08:30",
        ),
        Task(
            task_id="TASK-108",
            title="Add outlier detection using 90th percentile threshold",
            assignee="Sujal",
            status=TaskStatus.TODO,
            priority=Priority.LOW,
            category="Analytics",
            estimated_hours=2.5,
            actual_hours=0.0,
            tags="numpy, outliers, math",
            created_at="2026-10-14 09:00",
        ),
        Task(
            task_id="TASK-109",
            title="Write comprehensive unit tests for state machine",
            assignee="Member3",
            status=TaskStatus.TODO,
            priority=Priority.HIGH,
            category="Testing",
            estimated_hours=4.0,
            actual_hours=0.0,
            tags="testing, pytest, quality",
            created_at="2026-10-14 10:00",
        ),
        Task(
            task_id="TASK-110",
            title="Resolve third-party library dependency conflict",
            assignee="Meet",
            status=TaskStatus.BLOCKED,
            priority=Priority.CRITICAL,
            category="DevOps",
            estimated_hours=2.0,
            actual_hours=3.0,
            tags="bug, devops, blocking",
            created_at="2026-10-14 10:30",
        ),
    ]

    for t in tasks:
        engine.add_task(t)
    print(f"Successfully seeded {len(tasks)} sample tasks into {engine.storage.file_path}")


if __name__ == "__main__":
    eng = KanbanEngine("tasks.csv")
    populate_sample_data(eng)
