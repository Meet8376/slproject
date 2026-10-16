# 📋 Software Lab Development Project - Project Report

**Date of Report:** October 16, 2026 - 08:27  
**Core Technologies:** Python 3, Pandas (DataFrames & Vectorization), NumPy (Statistical Computations)  
**Academic Course:** Software Lab I (SL)

---

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Total Registered Tasks** | 11 |
| **Completed Tasks (DONE)** | 3 |
| **Completion Rate** | **27.27%** |
| **In Progress Tasks** | 4 |
| **Backlog (TODO)** | 2 |
| **Blocked Items** | 1 |
| **Total Estimated Effort** | 59.5 hrs |
| **Total Actual Effort** | 31.5 hrs |

---

## 2. Statistical Analysis & Effort Distribution (NumPy Engine)

This section leverages **NumPy vectorization and statistical distributions** (as implemented in Practical 5):

* **Mean Task Effort:** `2.86` hours
* **Standard Deviation (sigma):** `2.1` hours
* **Median Effort (50th Percentile):** `3.0` hours
* **75th Percentile:** `4.5` hours
* **90th Percentile:** `5.5` hours
* **Estimation Variance:** Mean difference of `-2.55` hours between actual and estimated effort.
* **Underestimated Tasks Count:** `3` tasks
* **Detected Effort Outliers:** `None`

---

## 3. Team Member Workload Breakdown

| assignee   |   total_tasks |   done_tasks |   in_progress |   estimated_hours |   actual_hours |   completion_rate_pct |
|:-----------|--------------:|-------------:|--------------:|------------------:|---------------:|----------------------:|
| Meet       |             4 |            1 |             1 |              15.5 |           13.5 |                  25   |
| Member3    |             3 |            1 |             1 |              11.5 |            6   |                  33.3 |
| Sujal      |             3 |            1 |             1 |              12.5 |           10   |                  33.3 |
| Naman      |             1 |            0 |             1 |              20   |            2   |                   0   |

## 4. Engineering Category Distribution

| category   |   task_count |   total_hours |   avg_hours |
|:-----------|-------------:|--------------:|------------:|
| Analytics  |            2 |           6   |         3   |
| Backend    |            2 |           9.5 |         4.8 |
| CLI        |            1 |           3.5 |         3.5 |
| COre       |            1 |           2   |         2   |
| Core       |            1 |           5   |         5   |
| DevOps     |            1 |           3   |         3   |
| Reporting  |            1 |           2.5 |         2.5 |
| Storage    |            1 |           0   |         0   |
| Testing    |            1 |           0   |         0   |
## 5. Risk & Escalation Register

### ⚠️ Blocked Tasks Requiring Attention
- **[TASK-110] Resolve third-party library dependency conflict** (Assignee: @Meet)

### 🚨 Critical Priority Tasks
- **[TASK-104] Add state transition validation & constraint rules** - Status: `REVIEW` (Assignee: @Meet)
- **[TASK-110] Resolve third-party library dependency conflict** - Status: `BLOCKED` (Assignee: @Meet)

## 6. 🤖 Machine Learning Predictive Insights (NumPy Engine)

Trained on historical task outcomes via **Multiple Linear Regression** and **Logistic Regression** (Gradient Descent):

* **Effort Predictor R² Score:** `0.762` (Mean Absolute Error: `0.54` hrs)
* **Risk Classifier Accuracy:** `90.0%` (Precision: `85.7%`)

### Key Influencing Features (Learned Weights):
- **`estimated_hours`**: Weight `+1.179` (increases effort)
- **`assignee_historical_bias`**: Weight `+1.072` (increases effort)
- **`tags_count`**: Weight `+0.537` (increases effort)
- **`priority_level`**: Weight `+0.377` (increases effort)

### Active Task Predictive Audit:
| task_id   | title                           | assignee   | status      |   estimated_hrs |   predicted_hrs | variance   | overrun_risk     |
|:----------|:--------------------------------|:-----------|:------------|----------------:|----------------:|:-----------|:-----------------|
| TASK-104  | Add state transition validat... | Meet       | REVIEW      |             4.5 |            5.05 | +0.6h      | 🟡 MEDIUM (42.7%) |
| TASK-105  | Implement bulk DataFrame boo... | Sujal      | IN_PROGRESS |             5   |            4.96 | -0.0h      | 🟢 LOW (15.7%)    |
| TASK-106  | Build automated Markdown & C... | Member3    | IN_PROGRESS |             3.5 |            2.59 | -0.9h      | 🟢 LOW (0.3%)     |
| TASK-107  | Support Excel (.xlsx) storag... | Meet       | IN_PROGRESS |             3   |            3.73 | +0.7h      | 🟢 LOW (1.6%)     |
| TASK-108  | Add outlier detection using ... | Sujal      | TODO        |             2.5 |            2.91 | +0.4h      | 🟢 LOW (0.1%)     |
| TASK-109  | Write comprehensive unit tes... | Member3    | TODO        |             4   |            3.24 | -0.8h      | 🟢 LOW (3.7%)     |
| TASK-112  | CN ka ppt complete karna        | Naman      | IN_PROGRESS |            20   |            2.78 | -17.2h     | 🟢 LOW (0.0%)     |

---
*Generated automatically by Python Terminal Kanban Engine.*