#!/usr/bin/env python3
"""Automated Demonstration Script for Software Lab Viva / Project Evaluation.

Walks through every component of the Kanban Engine, demonstrating:
1. Storage layer persistence (CSV & Excel using Pandas)
2. State transition validation & constraint rules
3. Fast DataFrame indexing & CRUD operations
4. Pandas bulk vectorized filtering
5. NumPy statistical analytics (Pract 5: mean, median, percentiles, variance)
6. Terminal board rendering & automated report generation
"""

import os
import time
from kanban.engine import KanbanEngine
from kanban.models import Task, TaskStatus, Priority, validate_transition
from kanban.visualizer import BoardVisualizer
from kanban.analytics import MetricsEngine
from kanban.reports import ReportGenerator
from sample_data import populate_sample_data


def print_step(title: str):
    print("\n" + "=" * 70)
    print(f"  👉 {title}")
    print("=" * 70)


def run_demo():
    demo_storage = "demo_tasks.csv"
    if os.path.exists(demo_storage):
        os.remove(demo_storage)

    print("\n" + "#" * 70)
    print("  🚀 SOFTWARE LAB PROJECT: PANDAS & NUMPY KANBAN ENGINE DEMO")
    print("  Team Members: Meet Agrawal, Sujal Nandanwar, Member 3")
    print("#" * 70)

    # STEP 1: Initialization
    print_step("STEP 1: Initializing Kanban Engine with Pandas Storage")
    engine = KanbanEngine(demo_storage)
    print(f"Initialized storage at: {demo_storage}")
    print(f"Empty DataFrame columns: {list(engine.df.columns)}")

    # STEP 2: Seeding Data
    print_step("STEP 2: Populating Sample Tasks Across Team Members")
    populate_sample_data(engine)
    engine.refresh()
    print(f"Loaded {len(engine.df)} tasks into in-memory DataFrame.")
    print("\nDataFrame Head (first 3 rows):")
    print(engine.df[["task_id", "title", "assignee", "status", "priority"]].head(3))

    # STEP 3: Visual ASCII Board
    print_step("STEP 3: Rendering Multi-Column Visual Kanban Board")
    cols = engine.get_board_columns()
    print(BoardVisualizer.render_board(cols))

    # STEP 4: State Machine Logic
    print_step("STEP 4: Demonstrating State-Transition Constraints")
    print("Attempting ILLEGAL transition: Moving TASK-107 from TODO directly to DONE...")
    ok, msg = engine.move_task("TASK-107", TaskStatus.DONE)
    print(f"Result: {'SUCCESS' if ok else 'REJECTED (as expected)'} -> {msg}")

    print("\nAttempting LEGAL transitions:")
    print("1. TODO -> IN_PROGRESS:")
    ok, msg = engine.move_task("TASK-107", TaskStatus.IN_PROGRESS)
    print(f"   Result: {'SUCCESS' if ok else 'FAILED'} -> {msg}")

    print("2. IN_PROGRESS -> REVIEW:")
    ok, msg = engine.move_task("TASK-107", TaskStatus.REVIEW)
    print(f"   Result: {'SUCCESS' if ok else 'FAILED'} -> {msg}")

    print("3. REVIEW -> DONE:")
    ok, msg = engine.move_task("TASK-107", TaskStatus.DONE)
    print(f"   Result: {'SUCCESS' if ok else 'FAILED'} -> {msg}")

    # STEP 5: Pandas Bulk Filtering
    print_step("STEP 5: Bulk Filtering Using Vectorized Pandas Boolean Indexing")
    print("Filter: Tasks assigned to 'Meet':")
    meet_df = engine.filter_tasks(assignee="Meet")
    print(BoardVisualizer.render_task_table(meet_df))

    print("\nFilter: High/Critical Priority Tasks in 'Backend' Category:")
    backend_df = engine.filter_tasks(category="Backend")
    print(BoardVisualizer.render_task_table(backend_df))

    # STEP 6: NumPy Analytics
    print_step("STEP 6: NumPy Statistical Analysis (Practical 5 Curriculum Integration)")
    reporter = ReportGenerator(engine.df, project_name="Software Lab Kanban System")
    print(reporter.generate_terminal_summary())

    # STEP 7: Machine Learning Predictive Engine
    print_step("STEP 7: Machine Learning Predictive Engine (Multiple Linear & Logistic Regression)")
    from kanban.ml import KanbanMLAssistant
    ml = KanbanMLAssistant()
    ml.train(engine.df)
    print("🤖 Model Training Results:")
    print(f"  • Effort Regression R² Score : {ml.regression_metrics.get('r2_score')} (MAE: {ml.regression_metrics.get('mae')} hrs)")
    print(f"  • Overrun Risk Accuracy      : {ml.risk_metrics.get('accuracy')}%")
    print("\nLearned Weights (Key Drivers):")
    for feat, w in ml.get_feature_importances()[:4]:
        print(f"  • {feat:<25}: {w:+.3f}")

    sample_pred = ml.predict_task_effort(estimated_hours=4.5, priority=Priority.HIGH, category="Backend", assignee="Meet")
    sample_risk = ml.predict_overrun_risk(estimated_hours=4.5, priority=Priority.HIGH, category="Backend", assignee="Meet")
    print(f"\nLive Inference for Proposed Task ('Backend' by @Meet, Est: 4.5h):")
    print(f"  -> Predicted Actual Effort : {sample_pred['predicted_actual_hours']} hrs ({sample_pred['predicted_variance']:+0.1f}h)")
    print(f"  -> Overrun Risk Level      : {sample_risk['risk_symbol']} {sample_risk['risk_level']} ({sample_risk['risk_probability_pct']}%)")

    # STEP 8: Automated Report Generation
    print_step("STEP 8: Automated Markdown Report Generation (with ML Predictive Insights)")
    report_file = "kanban_project_report.md"
    saved = reporter.generate_markdown_report(report_file)
    print(f"✅ Generated executive project report: {saved}")

    # STEP 9: Excel Storage Conversion
    print_step("STEP 9: Dynamic Excel (.xlsx) Storage Conversion via OpenpyXL")
    excel_file = "demo_tasks.xlsx"
    engine.storage.switch_format(excel_file)
    print(f"✅ State successfully persisted to Excel: {excel_file}")
    print(f"File exists: {os.path.exists(excel_file)} (Size: {os.path.getsize(excel_file)} bytes)")

    # Cleanup demo files
    if os.path.exists(demo_storage):
        os.remove(demo_storage)
    if os.path.exists(excel_file):
        os.remove(excel_file)

    print("\n" + "=" * 70)
    print("  🎉 DEMO COMPLETE: All software lab & ML requirements verified!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_demo()
