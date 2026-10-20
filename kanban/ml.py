"""Machine-Learning Engine for the Kanban system.

Trains a RandomForestRegressor to predict ``actual_hours`` from
task attributes (priority, category, assignee, estimated_hours).

Design notes
------------
- Accepts the engine's live DataFrame so it stays in sync with any
  task data source (CSV, Excel, etc.) without extra coupling.
- Feature encoding follows NumPy/Pandas idioms already used in the
  project (vectorised map, pd.get_dummies, np.array casts).
- Model persistence uses joblib (ships with scikit-learn) and saves
  to a ``models/`` directory inside the project root.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .models import Priority


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Columns used as input features for the model.
FEATURE_COLUMNS: List[str] = ["priority", "category", "assignee", "estimated_hours"]

#: The target column to predict.
TARGET_COLUMN: str = "actual_hours"

#: Default directory (relative to the project root) where models are saved.
DEFAULT_MODEL_DIR: str = "models"

#: File name written by :py:meth:`MLEngine.save_model`.
MODEL_FILENAME: str = "rf_actual_hours.joblib"

#: Minimum number of completed tasks required to train the model.
MIN_TRAINING_SAMPLES: int = 10


class MLEngine:
    """Predict ``actual_hours`` from task metadata using a Random Forest.

    Parameters
    ----------
    df:
        The live DataFrame from :class:`kanban.engine.KanbanEngine`.
        Only rows with ``actual_hours > 0`` (i.e., completed/started
        tasks) are used for training.
    model_dir:
        Directory path where trained models will be saved/loaded.
        Created automatically if it does not exist.
    n_estimators:
        Number of trees in the Random Forest.
    random_state:
        Seed for reproducibility.
    """

    # Priority ordinal mapping reuses the existing PRIORITY_WEIGHTS dict so
    # the same semantics are preserved across the whole codebase.
    _PRIORITY_MAP: Dict[str, int] = Priority.PRIORITY_WEIGHTS  # LOW=1 ... CRITICAL=4

    def __init__(
        self,
        df: pd.DataFrame,
        model_dir: str = DEFAULT_MODEL_DIR,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.df: pd.DataFrame = df.copy()
        self.model_dir: Path = Path(model_dir)
        self.n_estimators: int = n_estimators
        self.random_state: int = random_state

        # Internals - populated by train()
        self._model: Optional[RandomForestRegressor] = None
        self._label_encoders: Dict[str, LabelEncoder] = {}
        self._feature_names: List[str] = []
        self._is_trained: bool = False
        self._last_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """``True`` after a successful :py:meth:`train` call."""
        return self._is_trained

    @property
    def last_metrics(self) -> Dict[str, Any]:
        """Evaluation metrics from the most recent training run."""
        return dict(self._last_metrics)

    def train(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Fit the Random Forest on completed-task data.

        Parameters
        ----------
        test_size:
            Fraction of samples held out for evaluation (default 0.2).

        Returns
        -------
        dict
            Keys: ``n_train``, ``n_test``, ``mae_train``, ``mae_test``,
            ``r2_train``, ``r2_test``, ``feature_importances``.

        Raises
        ------
        ValueError
            If fewer than :py:data:`MIN_TRAINING_SAMPLES` usable rows
            exist in the DataFrame.
        """
        X, y = self._prepare_features(self.df)

        if len(X) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_TRAINING_SAMPLES} tasks with actual_hours > 0 "
                f"to train the model. Found: {len(X)}."
            )

        # Split into train / test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Build and fit the forest
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2,
        )
        self._model.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate
        metrics = self._evaluate(X_train, y_train, X_test, y_test)
        self._last_metrics = metrics
        return metrics

    def predict(self, tasks_df: pd.DataFrame) -> np.ndarray:
        """Predict ``actual_hours`` for one or more tasks.

        Parameters
        ----------
        tasks_df:
            A DataFrame (or slice of the engine's DataFrame) containing
            at least the columns in :py:data:`FEATURE_COLUMNS`.

        Returns
        -------
        np.ndarray
            Predicted actual_hours values, shape ``(n_samples,)``.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        X, _ = self._prepare_features(tasks_df, fit_encoders=False)
        return self._model.predict(X)

    def predict_single(
        self,
        priority: str,
        category: str,
        assignee: str,
        estimated_hours: float,
    ) -> float:
        """Convenience wrapper to predict for a single task's attributes.

        Parameters
        ----------
        priority:
            Task priority string (``'LOW'``, ``'MEDIUM'``, ``'HIGH'``,
            ``'CRITICAL'``).
        category:
            Task category string (e.g. ``'Backend'``, ``'Testing'``).
        assignee:
            Team member name.
        estimated_hours:
            Planned effort in hours.

        Returns
        -------
        float
            Predicted ``actual_hours`` (non-negative).
        """
        row = pd.DataFrame(
            [
                {
                    "priority": str(priority).strip().upper(),
                    "category": str(category).strip(),
                    "assignee": str(assignee).strip(),
                    "estimated_hours": float(estimated_hours),
                    TARGET_COLUMN: 0.0,  # placeholder - ignored during prediction
                }
            ]
        )
        predictions = self.predict(row)
        return float(max(0.0, predictions[0]))

    def get_feature_importances(self) -> pd.Series:
        """Return feature importances as a sorted Pandas Series.

        Returns
        -------
        pd.Series
            Index = feature name, values = importance score (0-1),
            sorted descending.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        importances = np.array(self._model.feature_importances_, dtype=float)
        return (
            pd.Series(importances, index=self._feature_names)
            .sort_values(ascending=False)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, filename: str = MODEL_FILENAME) -> Path:
        """Serialise the trained model and encoders to ``model_dir/filename``.

        Parameters
        ----------
        filename:
            Output file name inside :py:attr:`model_dir`.

        Returns
        -------
        Path
            Absolute path of the saved file.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError("Model is not trained. Call train() first.")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.model_dir / filename

        payload: Dict[str, Any] = {
            "model": self._model,
            "label_encoders": self._label_encoders,
            "feature_names": self._feature_names,
        }
        joblib.dump(payload, out_path, compress=3)
        return out_path.resolve()

    def load_model(self, filename: str = MODEL_FILENAME) -> None:
        """Load a previously saved model from ``model_dir/filename``.

        Parameters
        ----------
        filename:
            File name inside :py:attr:`model_dir` to load.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(
                f"No saved model found at '{model_path}'. "
                "Train and save a model first."
            )

        payload: Dict[str, Any] = joblib.load(model_path)
        self._model = payload["model"]
        self._label_encoders = payload["label_encoders"]
        self._feature_names = payload["feature_names"]
        self._is_trained = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit_encoders: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and target vector y from a DataFrame.

        Parameters
        ----------
        df:
            Source DataFrame. Must contain :py:data:`FEATURE_COLUMNS`.
            When used for training, must also contain
            :py:data:`TARGET_COLUMN`.
        fit_encoders:
            If ``True``, fit new LabelEncoder instances (training time).
            If ``False``, reuse those already fitted (inference time).

        Returns
        -------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
            Contains zeros when TARGET_COLUMN is absent or set to
            a placeholder.
        """
        work = df.copy()

        # Keep only rows where actual_hours > 0 during training
        if fit_encoders and TARGET_COLUMN in work.columns:
            work = work[work[TARGET_COLUMN].fillna(0.0) > 0.0].copy()

        # --- Ordinal encoding of priority (uses existing PRIORITY_WEIGHTS) ---
        work["priority_enc"] = (
            work["priority"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(self._PRIORITY_MAP)
            .fillna(0)
            .astype(float)
        )

        # --- Label encoding of categorical string columns ---
        cat_cols: List[str] = ["category", "assignee"]
        cat_encoded_cols: List[str] = []

        for col in cat_cols:
            enc_col = f"{col}_enc"
            cat_encoded_cols.append(enc_col)

            if fit_encoders:
                le = LabelEncoder()
                work[enc_col] = le.fit_transform(work[col].astype(str).str.strip())
                self._label_encoders[col] = le
            else:
                le = self._label_encoders.get(col)
                if le is None:
                    work[enc_col] = 0.0
                else:
                    # Map known labels; unseen values fall back to 0
                    known: Dict[str, int] = {
                        cls: idx for idx, cls in enumerate(le.classes_)
                    }
                    work[enc_col] = (
                        work[col]
                        .astype(str)
                        .str.strip()
                        .map(known)
                        .fillna(0)
                        .astype(float)
                    )

        # --- Numeric feature ---
        work["estimated_hours"] = (
            work["estimated_hours"].fillna(0.0).astype(float)
        )

        # Build ordered feature list and store for importance reporting
        feature_cols: List[str] = ["priority_enc"] + cat_encoded_cols + ["estimated_hours"]
        if fit_encoders:
            self._feature_names = ["priority"] + cat_cols + ["estimated_hours"]

        X: np.ndarray = work[feature_cols].values.astype(float)

        # Target vector
        if TARGET_COLUMN in work.columns:
            y: np.ndarray = work[TARGET_COLUMN].fillna(0.0).values.astype(float)
        else:
            y = np.zeros(len(X), dtype=float)

        return X, y

    def _evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Compute MAE and R^2 on both train and test splits."""
        assert self._model is not None

        pred_train: np.ndarray = self._model.predict(X_train)
        pred_test: np.ndarray = self._model.predict(X_test)

        importances_series = self.get_feature_importances()

        return {
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "mae_train": round(float(mean_absolute_error(y_train, pred_train)), 3),
            "mae_test": round(float(mean_absolute_error(y_test, pred_test)), 3),
            "r2_train": round(float(r2_score(y_train, pred_train)), 4),
            "r2_test": round(float(r2_score(y_test, pred_test)), 4),
            "feature_importances": importances_series.to_dict(),
        }
