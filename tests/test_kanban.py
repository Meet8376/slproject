"""Unit tests for the Lightweight Kanban Engine.

Validates:
- State machine transition rules
- Pandas DataFrame operations and filtering
- Storage persistence (CSV and Excel)
- NumPy statistical computations
"""

import os
import unittest
import numpy as np
import pandas as pd

from kanban.models import Task, TaskStatus, Priority, validate_transition
from kanban.storage import StorageEngine
from kanban.engine import KanbanEngine
from kanban.analytics import MetricsEngine
from kanban.reports import ReportGenerator


class TestKanbanEngine(unittest.TestCase):
    """Test suite covering core Kanban engine functionality."""

    def setUp(self):
        self.test_csv = "test_tasks.csv"
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        self.engine = KanbanEngine(self.test_csv)

    def tearDown(self):
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
        if os.path.exists("test_report.md"):
            os.remove("test_report.md")

    def test_state_transition_validation(self):
        """Verify strict state-machine enforcement."""
        # Allowed transition: TODO -> IN_PROGRESS
        valid, msg = validate_transition(TaskStatus.TODO, TaskStatus.IN_PROGRESS)
        self.assertTrue(valid)

        # Disallowed transition: TODO -> DONE (cannot skip IN_PROGRESS and REVIEW)
        invalid, msg = validate_transition(TaskStatus.TODO, TaskStatus.DONE)
        self.assertFalse(invalid)
        self.assertIn("Illegal transition", msg)

        # Allowed transition: IN_PROGRESS -> REVIEW
        valid, _ = validate_transition(TaskStatus.IN_PROGRESS, TaskStatus.REVIEW)
        self.assertTrue(valid)

        # Allowed transition: REVIEW -> DONE
        valid, _ = validate_transition(TaskStatus.REVIEW, TaskStatus.DONE)
        self.assertTrue(valid)

    def test_task_crud_and_indexing(self):
        """Test adding, fetching via indexing, updating, and deleting."""
        task = Task(
            task_id="TASK-201",
            title="Setup Continuous Integration",
            assignee="Meet",
            status=TaskStatus.TODO,
            priority=Priority.HIGH,
            estimated_hours=4.0,
            actual_hours=0.0,
            tags="ci, devops",
        )
        ok, msg = self.engine.add_task(task)
        self.assertTrue(ok)

        # Fetch via O(1) index
        retrieved = self.engine.get_task("TASK-201")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.title, "Setup Continuous Integration")
        self.assertEqual(retrieved.assignee, "Meet")

        # Update field
        ok, _ = self.engine.update_task("TASK-201", actual_hours=3.5)
        self.assertTrue(ok)
        self.assertEqual(self.engine.get_task("TASK-201").actual_hours, 3.5)

        # Delete
        ok, _ = self.engine.delete_task("TASK-201")
        self.assertTrue(ok)
        self.assertIsNone(self.engine.get_task("TASK-201"))

    def test_bulk_filtering_pandas(self):
        """Test vectorized boolean indexing with Pandas."""
        t1 = Task("TASK-1", "Task 1", "Meet", TaskStatus.TODO, Priority.HIGH, category="Backend", actual_hours=2.0)
        t2 = Task("TASK-2", "Task 2", "Sujal", TaskStatus.DONE, Priority.LOW, category="Frontend", actual_hours=6.0)
        t3 = Task("TASK-3", "Task 3", "Meet", TaskStatus.DONE, Priority.HIGH, category="Backend", actual_hours=4.0)

        self.engine.add_task(t1)
        self.engine.add_task(t2)
        self.engine.add_task(t3)

        # Filter by assignee
        meet_tasks = self.engine.filter_tasks(assignee="Meet")
        self.assertEqual(len(meet_tasks), 2)

        # Filter by status
        done_tasks = self.engine.filter_tasks(status=TaskStatus.DONE)
        self.assertEqual(len(done_tasks), 2)

        # Combined filter
        meet_done = self.engine.filter_tasks(assignee="Meet", status=TaskStatus.DONE)
        self.assertEqual(len(meet_done), 1)
        self.assertEqual(meet_done.iloc[0]["task_id"], "TASK-3")

    def test_numpy_statistical_analysis(self):
        """Verify NumPy percentile, mean, and variance calculations (Practical 5)."""
        hours = [2.0, 4.0, 4.0, 6.0, 8.0, 12.0]
        for idx, h in enumerate(hours):
            t = Task(f"TASK-{idx+1}", f"Task {idx+1}", "Meet", actual_hours=h, estimated_hours=4.0)
            self.engine.add_task(t)

        metrics = MetricsEngine(self.engine.df)
        stats = metrics.get_numpy_statistical_analysis()

        # Compare directly with NumPy reference functions
        np_arr = np.array(hours, dtype=float)
        self.assertAlmostEqual(stats["mean_actual_hours"], round(float(np.mean(np_arr)), 2))
        self.assertAlmostEqual(stats["median_actual_hours"], round(float(np.median(np_arr)), 2))
        self.assertAlmostEqual(stats["percentile_90"], round(float(np.percentile(np_arr, 90)), 2))

    def test_report_generation(self):
        """Verify automated report file generation."""
        t = Task("TASK-1", "Core Engine", "Meet", TaskStatus.DONE, Priority.HIGH, actual_hours=5.0)
        self.engine.add_task(t)

        generator = ReportGenerator(self.engine.df, project_name="Test Project")
        path = generator.generate_markdown_report("test_report.md")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("Test Project", content)
        self.assertIn("TASK-1", content)
        self.assertIn("Machine Learning Predictive Insights", content)

    def test_ml_linear_regression(self):
        """Verify NumPy multiple linear regression model training and convergence."""
        from kanban.ml import LinearRegressionModel
        # Generate synthetic linear data: y = 2*x1 + 3*x2 + 1.0 + noise
        rng = np.random.default_rng(42)
        X = rng.uniform(1.0, 10.0, size=(100, 2))
        y = 2.0 * X[:, 0] + 3.0 * X[:, 1] + 1.0 + rng.normal(0, 0.2, size=100)

        model = LinearRegressionModel(learning_rate=0.05, epochs=1500)
        model.fit(X, y)
        metrics = model.evaluate(X, y)

        self.assertGreater(metrics["r2_score"], 0.90)
        self.assertLess(metrics["mae"], 1.0)

        # Single prediction test
        pred = model.predict(np.array([[5.0, 5.0]]))
        expected = 2.0 * 5.0 + 3.0 * 5.0 + 1.0  # 26.0
        self.assertAlmostEqual(pred[0], expected, delta=1.5)

    def test_ml_logistic_regression(self):
        """Verify NumPy logistic regression binary classification."""
        from kanban.ml import LogisticRegressionModel
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, size=(100, 2))
        # Linearly separable with sigmoid boundary
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegressionModel(learning_rate=0.05, epochs=1500)
        model.fit(X, y)
        metrics = model.evaluate(X, y)

        self.assertGreaterEqual(metrics["accuracy"], 80.0)

    def test_kanban_ml_assistant(self):
        """Verify end-to-end Kanban ML Assistant feature extraction and prediction."""
        from kanban.ml import KanbanMLAssistant
        assistant = KanbanMLAssistant()
        assistant.train(self.engine.df)

        self.assertTrue(assistant.is_trained)
        self.assertGreater(assistant.training_sample_count, 0)

        pred_effort = assistant.predict_task_effort(
            estimated_hours=4.0,
            priority=Priority.HIGH,
            category="Backend",
            assignee="Meet",
            tags="database, sql",
            title="Setup database replica",
        )
        self.assertIn("predicted_actual_hours", pred_effort)
        self.assertGreater(pred_effort["predicted_actual_hours"], 0.0)

        pred_risk = assistant.predict_overrun_risk(
            estimated_hours=4.0,
            priority=Priority.HIGH,
            category="Backend",
            assignee="Meet",
        )
        self.assertIn("risk_level", pred_risk)
        self.assertIn(pred_risk["risk_level"], ["LOW", "MEDIUM", "HIGH"])


if __name__ == "__main__":
    unittest.main()

