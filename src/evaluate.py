"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


@dataclass
class EvaluationResult:
    """Container for common classification evaluation metrics."""

    accuracy: float
    roc_auc: float | None
    confusion_matrix: np.ndarray


def evaluate_classifier(model: Any, x_test: np.ndarray, y_test: np.ndarray) -> EvaluationResult:
    """Evaluate a classifier with accuracy, ROC AUC (if available), and confusion matrix."""
    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    roc_auc: float | None = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(x_test)[:, 1]
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except ValueError:
            roc_auc = None

    return EvaluationResult(accuracy=acc, roc_auc=roc_auc, confusion_matrix=cm)

