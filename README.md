# 📋 Lightweight Terminal Kanban Engine

> **A Python-based, command-line Kanban project management engine powered by Pandas and NumPy.**  
> Developed for **Software Lab I (SL)** at Shri Ramdeobaba College of Engineering and Management (RCOEM / RKNEC).

---

## 📌 Problem Statement

Managing small-scale development projects using heavy graphical interfaces (like Jira, Trello, or web dashboards) can slow down command-line-driven developer workflows. Additionally, flat-file task lists (like plain text notes or basic CSV files) lack structured filtering, sorting, and state-transition logic, leading to inconsistent task records and missing project metrics.

## 💡 Proposed Solution

We built a lightweight, fast, terminal-based Kanban engine in Python using **Pandas** as the core data processing layer and **NumPy** for statistical effort analysis:
1. **DataFrame Board State:** In-memory representation of tasks using Pandas DataFrames for fast $O(1)$ indexing by `task_id`.
2. **Dual File Persistence:** Directly loads, updates, and persists board state to structured **CSV** and **Excel (`.xlsx`)** files.
3. **State-Transition Validation:** Strict finite-state transition constraints (e.g., preventing jumping directly from `TODO` to `DONE`).
4. **Vectorized Bulk Filtering:** Filtering across multiple dimensions (assignee, status, priority, category, tags, effort ranges) via Pandas boolean indexing.
5. **NumPy Statistical Engine:** Automated estimation variance, cycle times, effort percentiles (50th, 75th, 90th), and effort outlier detection.
6. **NumPy Machine Learning Engine:** Multiple Linear Regression with Gradient Descent for task effort prediction, and Logistic Regression with Sigmoid activation for overrun & delay risk detection.
7. **Automated Reporting:** Executive Markdown reports and terminal dashboards generated on demand with ML predictive audits.

---

## 👥 3-Member Team Work Division

| Member | Role & Assigned Module | Key Contributions & Curriculum Links |
| :--- | :--- | :--- |
| **Meet Agrawal**<br>*(Roll No: 13, Class: A8)* | **Core Data Model & Persistence Layer** | • Designed `Task` class (OOP - Practical 4)<br>• Finite state-transition validation rules & sets (Practicals 2 & 3)<br>• `StorageEngine` for persistent CSV and Excel I/O via Pandas |
| **Sujal Nandanwar**<br>*(Roll No: 40, Class: A8)* | **Data Processing & Statistical Analytics** | • Pandas DataFrame indexing and vectorized boolean filtering (`filter_tasks`)<br>• NumPy analytics engine (`MetricsEngine` - Practical 5): mean, median, standard deviation, 90th percentile, estimation variance |
| **Team Member 3** | **Visualizer, CLI & Machine Learning** | • Multi-column ASCII terminal Kanban board (`BoardVisualizer`)<br>• Pure NumPy learning algorithms (`LinearRegressionModel`, `LogisticRegressionModel`, `KanbanMLAssistant`)<br>• `ReportGenerator` for automated Markdown reports & terminal summaries<br>• Interactive menu loop and CLI argument parser (`kanban_cli.py`) |

---

## 📚 Curriculum Integration: How Class Concepts are Applied

| Syllabus Concept | Practical Reference | Implementation in this Project |
| :--- | :--- | :--- |
| **Lists, Tuples & Sets** | **Practical 2** | Tag deduplication, sorting priorities, tuple-based range queries, and sets for allowed state transitions. |
| **Dictionaries & Nested Data** | **Practical 3** | Mapping valid transitions (`VALID_TRANSITIONS`), task dictionaries, status counts, and column grouping dictionaries. |
| **OOP & File I/O** | **Practical 4** | `Task`, `StorageEngine`, `KanbanEngine` classes; file persistence with safe schema verification and delimiter handling. |
| **NumPy Array Computations** | **Practical 5** | Effort analysis via `np.mean()`, `np.median()`, `np.std()`, `np.var()`, `np.percentile()`, boolean indexing for outlier detection (`actual_hours >= threshold`), and estimation error vectors. |
| **Pandas DataFrames** | **Advanced SL** | `pd.read_csv`, `to_csv`, `pd.read_excel`, fast `.loc[task_id]` indexing, multi-column `.groupby()`, and crosstabs. |
| **Machine Learning / Statistical Learning** | **SL Advanced Project** | Vectorized matrix math ($X \mathbf{w} + b$), Z-score standardization, Batch Gradient Descent, MSE & Cross-Entropy loss functions, $R^2$ evaluation, and Sigmoid probability classification built purely from scratch using NumPy. |

---

## 📁 Repository Structure

```text
slproject/
├── .gitignore               # Excludes .venv and Python cache
├── requirements.txt         # Project dependencies (pandas, numpy, openpyxl, tabulate)
├── README.md                # Comprehensive project documentation
├── tasks.csv                # Persistent CSV storage (auto-created if missing)
├── sample_data.py           # Seeder script with realistic sample tasks
├── demo.py                  # Step-by-step automated demo for lab viva
├── kanban_cli.py            # Main application (Interactive Menu + CLI commands)
├── tests/
│   └── test_kanban.py       # Unit tests covering state machine, filtering, stats & ML
└── kanban/
    ├── __init__.py          # Package exports
    ├── models.py            # Task model, Status, Priority, State-transition rules
    ├── storage.py           # CSV and Excel persistence engine using Pandas
    ├── engine.py            # DataFrame indexing, CRUD operations, bulk filtering
    ├── analytics.py         # NumPy statistical engine & Pandas aggregations
    ├── ml.py                # Pure NumPy Linear & Logistic Regression learning engine
    ├── visualizer.py        # ASCII multi-column terminal board and tables
    └── reports.py           # Markdown report and terminal dashboard generator
```

---

## 🚀 Getting Started

### 1. Prerequisites & Virtual Environment

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/Meet8376/slproject.git
cd slproject

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Mode A: Interactive Terminal Menu

Run without arguments to launch the full interactive terminal application:

```bash
python kanban_cli.py
```

Features in Interactive Mode:
- **[1] View Kanban Board:** Live multi-column ASCII Kanban visualization.
- **[2] Add New Task:** Prompted task creation with priority, estimated hours, and tags.
- **[3] Move Task:** Moves task through workflow with state-machine constraint validation.
- **[4] Update Task:** Modify assignee, title, priority, or log actual hours.
- **[5] Delete Task:** Safe task removal.
- **[6] Filter Tasks:** Vectorized Pandas search across assignee, priority, status, or tag.
- **[7] Sort Tasks:** Sort by priority weight, actual hours, or estimated hours.
- **[8] NumPy Analytics:** View statistical distribution (mean, standard deviation, 90th percentile).
- **[9] Generate Report:** Generates an executive Markdown report (`project_report.md`).
- **[10] Storage Management:** Convert storage seamlessly between CSV and Excel (`tasks.xlsx`), or create backups.

---

### Mode B: Command-Line Interface (CLI)

The tool can also be called directly from terminal scripts:

```bash
# 1. View the visual Kanban board
python kanban_cli.py board

# 2. Add a new task
python kanban_cli.py add --title "Refactor database query" --assignee "Meet" --priority HIGH --estimated 3.5

# 3. Move task status (validates state transition)
python kanban_cli.py move TASK-107 IN_PROGRESS

# 4. Filter tasks using Pandas boolean indexing
python kanban_cli.py filter --assignee Meet --status IN_PROGRESS

# 5. Display NumPy and Pandas project health summary
python kanban_cli.py stats

# 6. Generate an executive Markdown report
python kanban_cli.py report --output project_report.md
```

---

### Mode C: Automated Lab Viva Walkthrough

To demonstrate all features sequentially during a practical viva or project review, run:

```bash
python demo.py
```

This runs a step-by-step execution:
1. Initializes storage and loads Pandas DataFrame.
2. Seeds tasks across group members.
3. Renders the ASCII Kanban board.
4. Demonstrates state-machine validation (rejection of illegal transition vs acceptance of valid flow).
5. Executes Pandas bulk filtering.
6. Computes NumPy metrics (mean, median, 90th percentile, estimation variance).
7. Exports Markdown project reports.
8. Demonstrates dynamic Excel conversion.

---

## 🧪 Running Unit Tests

Run the test suite using Python's built-in `unittest` runner:

```bash
python -m unittest discover tests
```

Tests verify:
- ✅ State machine constraint validation (`validate_transition`)
- ✅ Task CRUD operations and Pandas `.loc` indexing
- ✅ Pandas vectorized boolean filtering
- ✅ NumPy statistical functions against reference formulas
- ✅ Markdown report export integrity

---

## 📊 Sample Board Output

```text
================================================================================
=============== 📋 TERMINAL KANBAN BOARD (Pandas & NumPy Engine) ================
================================================================================

─────── TODO (3) ─────── | ─── IN_PROGRESS (2) ──── | ────── REVIEW (1) ────── | ─────── DONE (3) ─────── | ───── BLOCKED (1) ──────
────────────────────────────────────────────────────────────────────────────────
[TASK-107] MEDI          │ [TASK-105] HIGH          │ [TASK-104] CRIT          │ [TASK-101] HIGH          │ [TASK-110] CRIT         
 Support Excel (.xls...  │  Implement bulk Data...  │  Add state transitio...  │  Design Pandas schem...  │  Resolve third-party... 
 @Meet | 0.0h/3.0h       │  @Sujal | 4.0h/5.0h      │  @Meet | 5.0h/4.5h       │  @Meet | 5.5h/6.0h       │  @Meet | 3.0h/2.0h      
────────────────────────────────────────────────────────────────────────────────
[TASK-108] LOW           │ [TASK-106] MEDI          │                          │ [TASK-102] HIGH          │                         
 Add outlier detecti...  │  Build automated Mar...  │                          │  Implement NumPy sta...  │                         
 @Sujal | 0.0h/2.5h      │  @Member3 | 2.5h/3.5h    │                          │  @Sujal | 6.0h/5.0h      │                         
────────────────────────────────────────────────────────────────────────────────
```

---

## 📜 License & Acknowledgements

Developed as part of the **Software Lab I** course curriculum.  
Uses Python standard libraries along with **NumPy**, **Pandas**, **OpenpyXL**, and **Tabulate**.