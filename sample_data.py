"""Sample dataset generator for the Kanban Engine.

Provides two public helpers:
- ``populate_sample_data``  – seeds 10 hand-crafted showcase tasks.
- ``generate_synthetic_tasks`` – yields 300 synthetic Task objects
  with realistic hour patterns for ML training (kanban.ml.MLEngine).
"""

import random
from typing import Generator, Optional

import numpy as np

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


# ---------------------------------------------------------------------------
# Synthetic data generator for ML training
# ---------------------------------------------------------------------------

# Realistic per-category effort ranges (min_hours, max_hours)
_CATEGORY_HOUR_RANGES = {
    "Backend":    (3.0, 10.0),
    "Frontend":   (2.0,  8.0),
    "Analytics":  (2.5,  9.0),
    "Testing":    (1.5,  6.0),
    "DevOps":     (2.0,  8.0),
    "CLI":        (1.5,  5.0),
    "Reporting":  (1.0,  4.0),
    "Storage":    (2.0,  7.0),
    "Core":       (3.0, 12.0),
    "Security":   (3.0, 10.0),
}

# Average over-run factor per priority level (actual = estimated * factor + noise)
_PRIORITY_OVERRUN = {
    Priority.LOW:      0.85,   # low-priority tasks often overestimated
    Priority.MEDIUM:   1.00,   # roughly on target
    Priority.HIGH:     1.15,   # slightly underestimated
    Priority.CRITICAL: 1.30,   # frequently runs over schedule
}

# Assignee-specific productivity modifier (< 1 = faster, > 1 = slower)
_ASSIGNEE_MODIFIER = {
    "Meet":    0.90,
    "Sujal":   1.05,
    "Member3": 1.10,
    "Priya":   0.95,
    "Arjun":   1.00,
}

_TASK_TITLE_TEMPLATES = [
    "Implement {feature} module",
    "Refactor {feature} layer",
    "Write unit tests for {feature}",
    "Optimise {feature} performance",
    "Debug {feature} pipeline",
    "Design {feature} schema",
    "Integrate {feature} API",
    "Document {feature} interface",
    "Review {feature} pull request",
    "Deploy {feature} to staging",
    "Fix regression in {feature}",
    "Add caching to {feature}",
    "Audit {feature} for security",
    "Benchmark {feature} throughput",
    "Migrate {feature} to new stack",
]

_FEATURES = [
    "authentication", "data pipeline", "reporting", "task scheduler",
    "notification service", "search index", "export engine",
    "analytics dashboard", "storage adapter", "REST client",
    "queue processor", "config loader", "metrics collector",
    "logging framework", "test harness", "CI workflow",
    "database migration", "API gateway", "caching layer", "state machine",
]

_TAG_POOL = [
    "pandas", "numpy", "api", "testing", "bug", "refactor",
    "performance", "security", "database", "ci-cd", "docs",
    "auth", "logging", "cache", "migration", "scheduler",
]


def generate_synthetic_tasks(
    n: int = 300,
    seed: Optional[int] = 7,
    start_task_id: int = 200,
) -> Generator[Task, None, None]:
    """Yield *n* synthetic :class:`~kanban.models.Task` objects for ML training.

    Each generated task has realistic correlations between priority,
    category, assignee and the resulting ``actual_hours``:

    - Higher-priority tasks tend to over-run their estimates.
    - Certain categories (Core, Security) have longer effort ranges.
    - Assignee productivity modifiers add inter-person variance.
    - Gaussian noise is added to every sample so the data is not
      perfectly linear (mimics real-world messiness).

    Parameters
    ----------
    n:
        Number of tasks to generate (default 300).
    seed:
        NumPy / random seed for reproducibility (default 7).
        Pass ``None`` for non-deterministic output.
    start_task_id:
        Numeric suffix of the first synthetic task ID (default 200,
        so IDs start at ``TASK-200`` and avoid collisions with the
        hand-crafted ``TASK-10x`` tasks).

    Yields
    ------
    Task
        One synthetic Task per iteration.  All yielded tasks have
        ``status=DONE`` and ``actual_hours > 0`` so they are immediately
        usable as training samples for :class:`~kanban.ml.MLEngine`.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    categories = list(_CATEGORY_HOUR_RANGES.keys())
    priorities = Priority.ALL_PRIORITIES
    assignees = list(_ASSIGNEE_MODIFIER.keys())

    for i in range(n):
        task_id = f"TASK-{start_task_id + i}"
        category = random.choice(categories)
        priority = random.choice(priorities)
        assignee = random.choice(assignees)

        # Estimated hours: draw from the category-specific range
        low, high = _CATEGORY_HOUR_RANGES[category]
        estimated = round(float(np.random.uniform(low, high)), 1)

        # Actual hours: estimated * priority_factor * assignee_modifier + noise
        priority_factor = _PRIORITY_OVERRUN[priority]
        assignee_mod = _ASSIGNEE_MODIFIER[assignee]
        noise = float(np.random.normal(loc=0.0, scale=0.5))
        actual = estimated * priority_factor * assignee_mod + noise
        actual = round(max(0.5, actual), 2)   # clamp to a positive value

        # Title & tags
        template = random.choice(_TASK_TITLE_TEMPLATES)
        feature = random.choice(_FEATURES)
        title = template.format(feature=feature)
        n_tags = random.randint(1, 3)
        tags = ", ".join(random.sample(_TAG_POOL, n_tags))

        yield Task(
            task_id=task_id,
            title=title,
            assignee=assignee,
            status=TaskStatus.DONE,
            priority=priority,
            category=category,
            estimated_hours=estimated,
            actual_hours=actual,
            tags=tags,
        )


def seed_synthetic_data(engine: KanbanEngine, n: int = 300, seed: Optional[int] = 7) -> int:
    """Load *n* synthetic tasks into *engine* and return how many were added.

    Skips any task whose ID already exists on the board so the function
    is safe to call on a non-empty engine.

    Parameters
    ----------
    engine:
        A live :class:`~kanban.engine.KanbanEngine` instance.
    n:
        Number of synthetic tasks to attempt (default 300).
    seed:
        RNG seed forwarded to :func:`generate_synthetic_tasks`.

    Returns
    -------
    int
        Count of tasks successfully added.
    """
    added = 0
    for task in generate_synthetic_tasks(n=n, seed=seed):
        ok, _ = engine.add_task(task)
        if ok:
            added += 1
    return added
