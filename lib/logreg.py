"""PLAN.md step 7: logistic regression reference for the evasion task (standardise + L2 logistic regression)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from lib import SEED


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float = 1.0, seed: int = SEED):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, random_state=seed)).fit(X, y)


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """P(evading) per row."""
    return model.predict_proba(X)[:, 1]
