"""Quick ML Engine demonstration script.

Shows: training, evaluation metrics, feature importances, and predictions.
Run with:  python -X utf8 ml_demo.py
"""

from sample_data import generate_synthetic_tasks
from kanban.ml import MLEngine
import pandas as pd

print("=" * 50)
print("       ML ENGINE DEMO  (kanban/ml.py)")
print("=" * 50)

# ── 1. Build training DataFrame ──────────────────────
print("\n[1] Generating 300 synthetic tasks...")
tasks = list(generate_synthetic_tasks(n=300, seed=7))
df = pd.DataFrame([t.to_dict() for t in tasks])
print(f"    DataFrame shape : {df.shape}")
print(f"    actual_hours    : mean={df['actual_hours'].mean():.2f}  std={df['actual_hours'].std():.2f}")

# ── 2. Train ──────────────────────────────────────────
print("\n[2] Training RandomForestRegressor (200 trees)...")
ml = MLEngine(df, model_dir="models")
metrics = ml.train()

print("\n    --- Evaluation Metrics ---")
print(f"    Training samples : {metrics['n_train']}")
print(f"    Test samples     : {metrics['n_test']}")
print(f"    MAE  (train)     : {metrics['mae_train']} hrs")
print(f"    MAE  (test)      : {metrics['mae_test']} hrs")
print(f"    R2   (train)     : {metrics['r2_train']}")
print(f"    R2   (test)      : {metrics['r2_test']}")

# ── 3. Feature importances ────────────────────────────
print("\n    --- Feature Importances ---")
for feat, score in metrics["feature_importances"].items():
    bar = "#" * int(score * 40)
    print(f"    {feat:<18} {score:.3f}  {bar}")

# ── 4. Single-task predictions ────────────────────────
print("\n[3] Single-Task Predictions")
print("    (priority, category, assignee, estimated_hrs) -> predicted actual_hrs\n")
cases = [
    ("CRITICAL", "Core",      "Meet",    8.0),
    ("HIGH",     "Backend",   "Sujal",   6.0),
    ("MEDIUM",   "Testing",   "Priya",   4.0),
    ("LOW",      "Reporting", "Member3", 2.0),
    ("CRITICAL", "Security",  "Arjun",  10.0),
]
for pri, cat, who, est in cases:
    pred = ml.predict_single(pri, cat, who, est)
    diff = pred - est
    sign = "+" if diff >= 0 else ""
    print(f"    [{pri:<8}] {cat:<10} @{who:<8}  est={est:.1f}h  =>  predicted={pred:.2f}h  ({sign}{diff:.2f}h)")

# ── 5. Save ───────────────────────────────────────────
saved = ml.save_model()
print(f"\n[4] Model saved  ->  {saved}")
print("\n" + "=" * 50)
print("    Done! Model is at  models/rf_actual_hours.joblib")
print("=" * 50)
