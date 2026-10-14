# 📋 Software Lab Kanban System - Project Report

**Date of Report:** October 14, 2026 - 15:57  
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

---
*Generated automatically by Python Terminal Kanban Engine.*