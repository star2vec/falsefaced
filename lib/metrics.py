"""Evaluation shared by all evasion classifiers (steps 6-8). Input: P(evading) for the held-out rows of a Task."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.chameleon import CONCEPTS
from lib.task import Task


def evaluate(task: Task, p_test: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """One row per concept plus 'all'. Columns:
    acc, auc            evading vs control (none + other) on the held-out rows
    auc_own_vs_other    only the triggered rows: can it tell *which* trigger, not just that there is one
    auc_own_vs_none     only untriggered controls
    n                   held-out rows"""
    m = task.meta[task.is_test].reset_index(drop=True)
    y = task.y[task.is_test]
    rows = []
    for name, sel in [(c, (m.concept == c).values) for c in CONCEPTS] + [("all", np.ones(len(m), bool))]:
        yy, pp, cond = y[sel], p_test[sel], m.condition.values[sel]
        trig = cond != "none"; untrig = cond != "other"
        rows.append(dict(concept=name, n=int(sel.sum()), acc=((pp > threshold) == yy).mean(), auc=roc_auc_score(yy, pp),
                         auc_own_vs_other=roc_auc_score(yy[trig], pp[trig]), auc_own_vs_none=roc_auc_score(yy[untrig], pp[untrig])))
    return pd.DataFrame(rows).set_index("concept")
