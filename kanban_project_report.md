# 📋 Software Lab Kanban System - Project Report

**Date of Report:** October 16, 2026 - 08:28  
**Core Technologies:** Python 3, Pandas (DataFrames & Vectorization), NumPy (Statistical Computations)  
**Academic Course:** Software Lab I (SL)

---

## 1. Executive Summary

| Metric | Value |
| :--- | :--- |
| **Total Registered Tasks** | 10 |
| **Completed Tasks (DONE)** | 4 |
| **Completion Rate** | **40.0%** |
| **In Progress Tasks** | 2 |
| **Backlog (TODO)** | 2 |
| **Blocked Items** | 1 |
| **Total Estimated Effort** | 39.5 hrs |
| **Total Actual Effort** | 29.5 hrs |

---

## 2. Statistical Analysis & Effort Distribution (NumPy Engine)

This section leverages **NumPy vectorization and statistical distributions** (as implemented in Practical 5):

* **Mean Task Effort:** `2.95` hours
* **Standard Deviation (sigma):** `2.18` hours
* **Median Effort (50th Percentile):** `3.25` hours
* **75th Percentile:** `4.75` hours
* **90th Percentile:** `5.55` hours
* **Estimation Variance:** Mean difference of `-1.0` hours between actual and estimated effort.
* **Underestimated Tasks Count:** `3` tasks
* **Detected Effort Outliers:** `None`

---

## 3. Team Member Workload Breakdown

| assignee   |   total_tasks |   done_tasks |   in_progress |   estimated_hours |   actual_hours |   completion_rate_pct |
|:-----------|--------------:|-------------:|--------------:|------------------:|---------------:|----------------------:|
| Meet       |             4 |            2 |             0 |              15.5 |           13.5 |                  50   |
| Member3    |             3 |            1 |             1 |              11.5 |            6   |                  33.3 |
| Sujal      |             3 |            1 |             1 |              12.5 |           10   |                  33.3 |

## 4. Engineering Category Distribution

| category   |   task_count |   total_hours |   avg_hours |
|:-----------|-------------:|--------------:|------------:|
| Analytics  |            2 |           6   |         3   |
| Backend    |            2 |           9.5 |         4.8 |
| CLI        |            1 |           3.5 |         3.5 |
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

* **Effort Predictor R² Score:** `0.923` (Mean Absolute Error: `0.33` hrs)
* **Risk Classifier Accuracy:** `89.7%` (Precision: `85.7%`)

### Key Influencing Features (Learned Weights):
- **`estimated_hours`**: Weight `+1.157` (increases effort)
- **`assignee_historical_bias`**: Weight `+0.493` (increases effort)
- **`category_id`**: Weight `+0.302` (increases effort)
- **`priority_level`**: Weight `+0.250` (increases effort)

### Active Task Predictive Audit:
| task_id   | title                           | assignee   | status      |   estimated_hrs |   predicted_hrs | variance   | overrun_risk     |
|:----------|:--------------------------------|:-----------|:------------|----------------:|----------------:|:-----------|:-----------------|
| TASK-104  | Add state transition validat... | Meet       | REVIEW      |             4.5 |            4.58 | +0.1h      | 🟡 MEDIUM (35.9%) |
| TASK-105  | Implement bulk DataFrame boo... | Sujal      | IN_PROGRESS |             5   |            5.04 | +0.0h      | 🟢 LOW (13.7%)    |
| TASK-106  | Build automated Markdown & C... | Member3    | IN_PROGRESS |             3.5 |            2.63 | -0.9h      | 🟢 LOW (0.1%)     |
| TASK-108  | Add outlier detection using ... | Sujal      | TODO        |             2.5 |            1.88 | -0.6h      | 🟢 LOW (0.1%)     |
| TASK-109  | Write comprehensive unit tes... | Member3    | TODO        |             4   |            3.54 | -0.5h      | 🟢 LOW (1.6%)     |

---
*Generated automatically by Python Terminal Kanban Engine.*