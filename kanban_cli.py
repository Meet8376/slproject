#!/usr/bin/env python3
"""Main Entrypoint for the Lightweight Terminal Kanban Engine.

Supports both an Interactive Menu Mode and a direct Command-Line Mode (CLI).
Developed for Software Lab Project using Python, Pandas, and NumPy.
"""

import sys
import argparse
from typing import Optional

from kanban.engine import KanbanEngine
from kanban.models import Task, TaskStatus, Priority
from kanban.visualizer import BoardVisualizer
from kanban.analytics import MetricsEngine
from kanban.reports import ReportGenerator
from kanban.ml import KanbanMLAssistant
from sample_data import populate_sample_data


def show_interactive_menu(engine: KanbanEngine) -> None:
    """Run an interactive CLI menu loop."""
    while True:
        # Display board
        cols = engine.get_board_columns()
        print(BoardVisualizer.render_board(cols))

        print("\n" + "─" * 60)
        print("  🧭 KANBAN ACTIONS MENU")
        print("─" * 60)
        print(" [1] 📋 View Kanban Board")
        print(" [2] ➕ Add New Task")
        print(" [3] ➡️  Move Task (State-Transition Logic)")
        print(" [4] ✏️  Update Task Details (Assignee, Hours, etc.)")
        print(" [5] 🗑️  Delete Task")
        print(" [6] 🔍 Filter Tasks (Assignee, Priority, Status, Tag)")
        print(" [7] 🔢 Sort Tasks (By Priority or Hours)")
        print(" [8] 📈 View NumPy Analytics & Effort Distribution")
        print(" [9] 📄 Generate Automated Project Report (Markdown)")
        print(" [10] 💾 Storage Management (Switch CSV / Excel / Backup)")
        print(" [11] ⚡ Seed Sample Demonstration Data")
        print(" [12] 🤖 Machine Learning Predictive Engine (Effort & Overrun Risk)")
        print(" [0] 🚪 Exit")
        print("─" * 60)

        choice = input("Enter option (0-12): ").strip()

        if choice == "0":
            print("\nExiting Kanban Engine. Board state saved.")
            break

        elif choice == "1":
            continue

        elif choice == "2":
            print("\n--- Add New Task ---")
            title = input("Enter Task Title: ").strip()
            if not title:
                print("Error: Task title cannot be empty.")
                continue

            assignee = input("Enter Assignee (default: Meet): ").strip() or "Meet"
            print(f"Priorities: {', '.join(Priority.ALL_PRIORITIES)}")
            priority = input("Enter Priority (default: MEDIUM): ").strip().upper() or Priority.MEDIUM
            if priority not in Priority.ALL_PRIORITIES:
                priority = Priority.MEDIUM

            category = input("Enter Category (e.g., Backend, Frontend, Core): ").strip() or "General"
            
            try:
                est_str = input("Estimated Hours (default: 2.0): ").strip() or "2.0"
                est_hours = float(est_str)
            except ValueError:
                est_hours = 2.0

            tags = input("Tags (comma-separated, e.g., 'ui, bug'): ").strip()

            # Machine Learning Inference on proposed task
            try:
                ml = KanbanMLAssistant()
                ml.train(engine.df)
                pred = ml.predict_task_effort(
                    estimated_hours=est_hours,
                    priority=priority,
                    category=category,
                    assignee=assignee,
                    tags=tags,
                    title=title,
                )
                risk = ml.predict_overrun_risk(
                    estimated_hours=est_hours,
                    priority=priority,
                    category=category,
                    assignee=assignee,
                    tags=tags,
                    title=title,
                )
                print("\n" + "─" * 45)
                print("  🤖 ML PREDICTION ASSISTANT (NumPy Models)")
                print("─" * 45)
                print(f"  • Predicted Actual Effort : {pred['predicted_actual_hours']} hrs ({pred['predicted_variance']:+0.1f}h from estimate)")
                print(f"  • Overrun Risk Level      : {risk['risk_symbol']} {risk['risk_level']} ({risk['risk_probability_pct']}% chance)")
                print(f"  • Velocity Factor         : {assignee} is {pred['assignee_pace_note']} ({pred['assignee_velocity_factor']}x)")
                print("─" * 45)
                adopt = input(f"Adopt ML-recommended effort ({pred['predicted_actual_hours']}h) instead of {est_hours}h? (y/N): ").strip().lower()
                if adopt == "y":
                    est_hours = pred["predicted_actual_hours"]
            except Exception as e:
                pass

            new_task = Task(
                task_id=engine.generate_next_task_id(),
                title=title,
                assignee=assignee,
                status=TaskStatus.TODO,
                priority=priority,
                category=category,
                estimated_hours=est_hours,
                actual_hours=0.0,
                tags=tags,
            )
            success, msg = engine.add_task(new_task)
            print(f"\n{'✅' if success else '❌'} {msg}")
            input("\nPress Enter to continue...")

        elif choice == "3":
            print("\n--- Move Task ---")
            task_id = input("Enter Task ID to move (e.g., TASK-101): ").strip().upper()
            task = engine.get_task(task_id)
            if not task:
                print(f"❌ Task '{task_id}' not found.")
                input("\nPress Enter to continue...")
                continue

            print(f"Current Status: {task.status}")
            print(f"Available Statuses: {', '.join(TaskStatus.ALL_STATUSES)}")
            target_status = input("Enter New Status: ").strip().upper()

            success, msg = engine.move_task(task_id, target_status)
            print(f"\n{'✅' if success else '❌'} {msg}")

            # If task was moved, offer to log actual hours
            if success and target_status in [TaskStatus.IN_PROGRESS, TaskStatus.REVIEW, TaskStatus.DONE]:
                hrs_input = input("Update actual hours logged on this task? (or press Enter to skip): ").strip()
                if hrs_input:
                    try:
                        engine.update_task(task_id, actual_hours=float(hrs_input))
                        print(f"Updated actual hours to {hrs_input} hrs.")
                    except ValueError:
                        pass
            input("\nPress Enter to continue...")

        elif choice == "4":
            print("\n--- Update Task ---")
            task_id = input("Enter Task ID: ").strip().upper()
            task = engine.get_task(task_id)
            if not task:
                print(f"❌ Task '{task_id}' not found.")
                input("\nPress Enter to continue...")
                continue

            print(f"Editing [{task.task_id}]: {task.title}")
            print("Leave blank to keep existing value.")
            new_title = input(f"Title [{task.title}]: ").strip()
            new_assignee = input(f"Assignee [{task.assignee}]: ").strip()
            new_priority = input(f"Priority [{task.priority}]: ").strip().upper()
            new_actual = input(f"Actual Hours [{task.actual_hours}]: ").strip()

            updates = {}
            if new_title:
                updates["title"] = new_title
            if new_assignee:
                updates["assignee"] = new_assignee
            if new_priority and new_priority in Priority.ALL_PRIORITIES:
                updates["priority"] = new_priority
            if new_actual:
                try:
                    updates["actual_hours"] = float(new_actual)
                except ValueError:
                    pass

            if updates:
                ok, msg = engine.update_task(task_id, **updates)
                print(f"{'✅' if ok else '❌'} {msg}")
            else:
                print("No changes made.")
            input("\nPress Enter to continue...")

        elif choice == "5":
            print("\n--- Delete Task ---")
            task_id = input("Enter Task ID to delete: ").strip().upper()
            confirm = input(f"Are you sure you want to delete {task_id}? (y/n): ").strip().lower()
            if confirm == "y":
                ok, msg = engine.delete_task(task_id)
                print(f"{'✅' if ok else '❌'} {msg}")
            else:
                print("Deletion cancelled.")
            input("\nPress Enter to continue...")

        elif choice == "6":
            print("\n--- Bulk Filtering (Pandas Vectorized Search) ---")
            status_f = input("Filter by Status (or Enter to skip): ").strip() or None
            assignee_f = input("Filter by Assignee (or Enter to skip): ").strip() or None
            priority_f = input("Filter by Priority (or Enter to skip): ").strip() or None
            category_f = input("Filter by Category (or Enter to skip): ").strip() or None
            tag_f = input("Filter by Tag (or Enter to skip): ").strip() or None

            filtered_df = engine.filter_tasks(
                status=status_f,
                assignee=assignee_f,
                priority=priority_f,
                category=category_f,
                tag=tag_f,
            )
            print("\n=== FILTERED RESULTS ===")
            print(BoardVisualizer.render_task_table(filtered_df))
            input("\nPress Enter to continue...")

        elif choice == "7":
            print("\n--- Sort Tasks ---")
            print("1. Sort by Priority (Critical -> Low)")
            print("2. Sort by Actual Hours (Descending)")
            print("3. Sort by Estimated Hours (Descending)")
            sort_opt = input("Choose sort option (1-3): ").strip()
            if sort_opt == "1":
                sorted_df = engine.sort_tasks(by="priority", ascending=False)
            elif sort_opt == "2":
                sorted_df = engine.sort_tasks(by="actual_hours", ascending=False)
            elif sort_opt == "3":
                sorted_df = engine.sort_tasks(by="estimated_hours", ascending=False)
            else:
                sorted_df = engine.df

            print("\n=== SORTED TASKS ===")
            print(BoardVisualizer.render_task_table(sorted_df))
            input("\nPress Enter to continue...")

        elif choice == "8":
            print("\n--- NumPy & Pandas Statistical Analytics (Practical 5) ---")
            reporter = ReportGenerator(engine.df)
            print(reporter.generate_terminal_summary())
            input("\nPress Enter to continue...")

        elif choice == "9":
            print("\n--- Generate Automated Project Report ---")
            file_name = input("Enter output markdown filename (default: project_report.md): ").strip() or "project_report.md"
            reporter = ReportGenerator(engine.df)
            saved_path = reporter.generate_markdown_report(file_name)
            if saved_path:
                print(f"✅ Full report generated successfully at: {saved_path}")
            else:
                print("❌ Failed to generate report.")
            input("\nPress Enter to continue...")

        elif choice == "10":
            print("\n--- Storage Management ---")
            print(f"Current Storage File: {engine.storage.file_path}")
            print(f"Format: {'Excel (.xlsx)' if engine.storage.is_excel else 'CSV'}")
            print("1. Create Timestamped Backup")
            print("2. Convert / Switch to Excel (tasks.xlsx)")
            print("3. Convert / Switch to CSV (tasks.csv)")
            s_opt = input("Choose option (1-3): ").strip()

            if s_opt == "1":
                b_path = engine.storage.create_backup()
                print(f"✅ Backup created: {b_path}")
            elif s_opt == "2":
                engine.storage.switch_format("tasks.xlsx")
                engine.refresh()
                print("✅ Storage switched to tasks.xlsx (Excel format).")
            elif s_opt == "3":
                engine.storage.switch_format("tasks.csv")
                engine.refresh()
                print("✅ Storage switched to tasks.csv (CSV format).")
            input("\nPress Enter to continue...")

        elif choice == "11":
            print("\n--- Seed Sample Data ---")
            confirm = input("Populate sample tasks for demonstration? (y/n): ").strip().lower()
            if confirm == "y":
                populate_sample_data(engine)
                engine.refresh()
            input("\nPress Enter to continue...")

        elif choice == "12":
            print("\n--- 🤖 Machine Learning Predictive Engine (NumPy Implementation) ---")
            ml = KanbanMLAssistant()
            ml.train(engine.df)

            reg_m = ml.regression_metrics
            risk_m = ml.risk_metrics
            weights = ml.get_feature_importances()

            print("\n📊 Model Performance Summary:")
            print(f"  • Multiple Linear Regression (Effort Predictor):")
            print(f"    - R² Score : {reg_m.get('r2_score', 0.0)} (Variance explained)")
            print(f"    - MAE      : {reg_m.get('mae', 0.0)} hours (Mean Absolute Error)")
            print(f"    - RMSE     : {reg_m.get('rmse', 0.0)} hours")
            print(f"  • Logistic Regression (Overrun Risk Classifier):")
            print(f"    - Accuracy : {risk_m.get('accuracy', 0.0)}%")
            print(f"    - Precision: {risk_m.get('precision', 0.0)}%")
            print(f"    - Recall   : {risk_m.get('recall', 0.0)}%")

            print("\n🧠 Learned Feature Weights (Key Drivers):")
            for feat_name, w in weights:
                direction = "increases effort" if w > 0 else "decreases effort"
                print(f"  • {feat_name:<26}: {w:+.3f} ({direction})")

            print("\n--- Interactive Task Simulator ---")
            sim = input("Simulate prediction for a proposed task? (y/n): ").strip().lower()
            if sim == "y":
                sim_title = input("Task Title: ").strip() or "Sample feature"
                sim_cat = input("Category (Backend/Analytics/Core/DevOps/CLI/Testing): ").strip() or "Backend"
                sim_prio = input("Priority (LOW/MEDIUM/HIGH/CRITICAL): ").strip().upper() or "MEDIUM"
                sim_ass = input("Assignee (Meet/Sujal/Member3/etc.): ").strip() or "Meet"
                try:
                    sim_est = float(input("Estimated Hours: ").strip() or "4.0")
                except ValueError:
                    sim_est = 4.0
                sim_tags = input("Tags (comma separated): ").strip()

                p_res = ml.predict_task_effort(sim_est, sim_prio, sim_cat, sim_ass, sim_tags, sim_title)
                r_res = ml.predict_overrun_risk(sim_est, sim_prio, sim_cat, sim_ass, sim_tags, sim_title)

                print("\n🎯 ML Inference Results:")
                print(f"  • Input Estimate           : {sim_est:.1f} hrs")
                print(f"  • Predicted Actual Effort  : {p_res['predicted_actual_hours']} hrs")
                print(f"  • Predicted Variance       : {p_res['predicted_variance']:+0.1f} hrs ({p_res['variance_percent']:+0.1f}%)")
                print(f"  • Overrun Risk Probability : {r_res['risk_symbol']} {r_res['risk_level']} ({r_res['risk_probability_pct']}%)")
                print(f"  • Assignee Pace Insight    : {sim_ass} is {p_res['assignee_pace_note']} ({p_res['assignee_velocity_factor']}x)")

            input("\nPress Enter to continue...")

        else:
            print("Invalid selection. Please choose a valid number.")
            input("\nPress Enter to continue...")


def run_cli_command(engine: KanbanEngine, args: argparse.Namespace) -> None:
    """Execute non-interactive CLI commands."""
    cmd = args.command

    if cmd == "board":
        cols = engine.get_board_columns()
        print(BoardVisualizer.render_board(cols))

    elif cmd == "add":
        new_task = Task(
            task_id=args.id or engine.generate_next_task_id(),
            title=args.title,
            assignee=args.assignee or "Unassigned",
            status=args.status or TaskStatus.TODO,
            priority=args.priority or Priority.MEDIUM,
            category=args.category or "General",
            estimated_hours=args.estimated or 0.0,
            actual_hours=args.actual or 0.0,
            tags=args.tags or "",
        )
        ok, msg = engine.add_task(new_task)
        print(f"{'SUCCESS' if ok else 'ERROR'}: {msg}")

    elif cmd == "move":
        ok, msg = engine.move_task(args.task_id, args.new_status)
        print(f"{'SUCCESS' if ok else 'ERROR'}: {msg}")

    elif cmd == "filter":
        df = engine.filter_tasks(
            status=args.status,
            assignee=args.assignee,
            priority=args.priority,
            category=args.category,
            tag=args.tag,
        )
        print(BoardVisualizer.render_task_table(df))

    elif cmd == "stats":
        reporter = ReportGenerator(engine.df)
        print(reporter.generate_terminal_summary())

    elif cmd == "report":
        file_path = args.output or "project_report.md"
        reporter = ReportGenerator(engine.df)
        path = reporter.generate_markdown_report(file_path)
        print(f"Report saved to: {path}")

    elif cmd == "seed":
        populate_sample_data(engine)
        print("Sample data populated.")

    elif cmd == "ml-eval":
        ml = KanbanMLAssistant()
        ml.train(engine.df)
        print("\n=== 🤖 MACHINE LEARNING MODEL EVALUATION (NumPy) ===")
        print(f"Training Samples: {ml.training_sample_count}")
        print(f"Linear Regression R² Score : {ml.regression_metrics.get('r2_score')}")
        print(f"Mean Absolute Error (MAE)  : {ml.regression_metrics.get('mae')} hrs")
        print(f"Root Mean Squared Error    : {ml.regression_metrics.get('rmse')} hrs")
        print(f"Risk Classifier Accuracy   : {ml.risk_metrics.get('accuracy')}%")
        print("\nFeature Weights:")
        for name, w in ml.get_feature_importances():
            print(f"  {name:<25}: {w:+.3f}")

    elif cmd == "ml-predict":
        ml = KanbanMLAssistant()
        ml.train(engine.df)
        p_res = ml.predict_task_effort(
            estimated_hours=args.estimated,
            priority=args.priority,
            category=args.category,
            assignee=args.assignee,
            tags=args.tags,
            title=args.title or "Proposed Task",
        )
        r_res = ml.predict_overrun_risk(
            estimated_hours=args.estimated,
            priority=args.priority,
            category=args.category,
            assignee=args.assignee,
            tags=args.tags,
            title=args.title or "Proposed Task",
        )
        print(f"\n🎯 Prediction for [{args.category}] task by @{args.assignee} (Estimate: {args.estimated}h):")
        print(f"  Predicted Actual Hours : {p_res['predicted_actual_hours']} hrs ({p_res['predicted_variance']:+0.1f}h)")
        print(f"  Overrun Risk Level     : {r_res['risk_symbol']} {r_res['risk_level']} ({r_res['risk_probability_pct']}%)")


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct argument parser for command line execution."""
    parser = argparse.ArgumentParser(
        description="Lightweight Terminal Kanban Engine in Python using Pandas & NumPy."
    )
    parser.add_argument("--storage", default="tasks.csv", help="Path to storage file (tasks.csv or tasks.xlsx)")

    subparsers = parser.add_subparsers(dest="command", help="Kanban Subcommands")

    # board
    subparsers.add_parser("board", help="Display visual Kanban board")

    # add
    add_parser = subparsers.add_parser("add", help="Add a new task")
    add_parser.add_argument("--title", required=True, help="Task title")
    add_parser.add_argument("--id", help="Optional Task ID (auto-generated if omitted)")
    add_parser.add_argument("--assignee", default="Unassigned", help="Task assignee")
    add_parser.add_argument("--priority", default=Priority.MEDIUM, help="Task priority")
    add_parser.add_argument("--status", default=TaskStatus.TODO, help="Initial status")
    add_parser.add_argument("--category", default="General", help="Task category")
    add_parser.add_argument("--estimated", type=float, default=0.0, help="Estimated hours")
    add_parser.add_argument("--actual", type=float, default=0.0, help="Actual hours")
    add_parser.add_argument("--tags", default="", help="Comma separated tags")

    # move
    move_parser = subparsers.add_parser("move", help="Move task state with validation")
    move_parser.add_argument("task_id", help="Task ID (e.g. TASK-101)")
    move_parser.add_argument("new_status", help="Target status (TODO, IN_PROGRESS, REVIEW, DONE, BLOCKED)")

    # filter
    filter_parser = subparsers.add_parser("filter", help="Filter tasks using Pandas")
    filter_parser.add_argument("--status", help="Filter by status")
    filter_parser.add_argument("--assignee", help="Filter by assignee")
    filter_parser.add_argument("--priority", help="Filter by priority")
    filter_parser.add_argument("--category", help="Filter by category")
    filter_parser.add_argument("--tag", help="Filter by tag")

    # stats
    subparsers.add_parser("stats", help="Compute NumPy & Pandas analytics")

    # report
    report_parser = subparsers.add_parser("report", help="Generate automated project report")
    report_parser.add_argument("--output", default="project_report.md", help="Output report file path")

    # seed
    subparsers.add_parser("seed", help="Seed sample dataset")

    # ml-eval
    subparsers.add_parser("ml-eval", help="Evaluate NumPy ML predictive models")

    # ml-predict
    pred_parser = subparsers.add_parser("ml-predict", help="Predict task duration and overrun risk")
    pred_parser.add_argument("--estimated", type=float, required=True, help="Estimated hours")
    pred_parser.add_argument("--priority", default=Priority.MEDIUM, help="Task priority")
    pred_parser.add_argument("--category", default="Backend", help="Task category")
    pred_parser.add_argument("--assignee", default="Meet", help="Assignee name")
    pred_parser.add_argument("--title", default="", help="Task title")
    pred_parser.add_argument("--tags", default="", help="Tags")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    engine = KanbanEngine(args.storage)

    if args.command:
        run_cli_command(engine, args)
    else:
        # If run without arguments, enter interactive menu mode
        try:
            show_interactive_menu(engine)
        except KeyboardInterrupt:
            print("\nExiting. Board state preserved.")
            sys.exit(0)


if __name__ == "__main__":
    main()
