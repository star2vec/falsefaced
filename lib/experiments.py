"""PLAN.md steps 8 and 9: experiments comparing naive Bayes, dependent Bayes (the researcher's code, via
lib.naive_bayes / lib.dependent_bayes) and the logistic reference, all on the rows and labels of lib.task.

Every classifier is a function (X_lab, y_lab, X_eval, X_unlab) -> P(evading) for the rows of X_eval.
X_unlab is the full set of training activations without labels: the binarisation thresholds (medians) come from
it, so with few labels the features are still the same; only the counting uses the labelled rows.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib import SEED, dependent_bayes as db, naive_bayes as nb
from lib.chameleon import CONCEPTS
from lib.logreg import fit_logreg, predict_proba
from lib.task import Task

warnings.filterwarnings("ignore", message="overflow encountered in exp")   # exp(+large) -> inf -> posterior 0, correct


def naive(X_lab, y_lab, X_eval, X_unlab):
    th = nb.medians(X_unlab)
    return nb.calcphx(nb.mapping(X_eval, th), nb.fit(nb.mapping(X_lab, th), y_lab))


def dependent(X_lab, y_lab, X_eval, X_unlab):
    """All pairs (K = all), the best setting in bayes.ipynb. Log-odds turned into a probability, same threshold."""
    th = nb.medians(X_unlab)
    x_lab = nb.mapping(X_lab, th)
    model = db.fit(x_lab, y_lab, db.sortcorrelation(x_lab, y_lab))
    return 1 / (1 + np.exp(-db.bayespredict(nb.mapping(X_eval, th), model)))


def logreg(X_lab, y_lab, X_eval, X_unlab):
    return predict_proba(fit_logreg(X_lab, y_lab), X_eval)


CLASSIFIERS = {"naive Bayes": naive, "dependent Bayes": dependent, "logistic regression": logreg}


def summary(y, p, condition) -> dict:
    """acc and AUC (evading vs control), and AUC among triggered rows only (own vs other)."""
    trig = condition != "none"
    return dict(acc=((p > 0.5) == y).mean(), auc=roc_auc_score(y, p), auc_own_vs_other=roc_auc_score(y[trig], p[trig]))


def full_data(task: Task) -> pd.DataFrame:
    Xtr, ytr = task.train; Xte, yte = task.test
    cond = task.meta.condition.values[task.is_test]
    return pd.DataFrame({name: summary(yte, f(Xtr, ytr, Xte, Xtr), cond) for name, f in CLASSIFIERS.items()}).T


def sample_labels(y: np.ndarray, n: int, rng) -> np.ndarray:
    """n training rows, at least one of each class, evading share as in the data (1/3)."""
    n_pos = max(1, int(round(n * y.mean()))); n_neg = n - n_pos
    return np.concatenate([rng.choice(np.where(y == 1)[0], n_pos, replace=False),
                           rng.choice(np.where(y == 0)[0], n_neg, replace=False)])


def label_curve(task: Task, sizes=(10, 20, 50, 100, None), repeats: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Held-out metrics when only n labelled training rows are used (None = all). `repeats` random draws per n."""
    Xtr, ytr = task.train; Xte, yte = task.test
    cond = task.meta.condition.values[task.is_test]
    rows = []
    for n in sizes:
        for r in range(1 if n is None else repeats):
            idx = np.arange(len(ytr)) if n is None else sample_labels(ytr, n, np.random.default_rng(seed + r))
            for name, f in CLASSIFIERS.items():
                rows.append(dict(n=len(idx), repeat=r, classifier=name,
                                 **summary(yte, f(Xtr[idx], ytr[idx], Xte, Xtr), cond)))
    return pd.DataFrame(rows)


def shuffled_null(task: Task, seed: int = SEED) -> pd.DataFrame:
    """Same as full_data but with the training labels permuted: every AUC should fall to ~0.5."""
    Xtr, ytr = task.train; Xte, yte = task.test
    cond = task.meta.condition.values[task.is_test]
    y_perm = np.random.default_rng(seed).permutation(ytr)
    return pd.DataFrame({name: summary(yte, f(Xtr, y_perm, Xte, Xtr), cond) for name, f in CLASSIFIERS.items()}).T


def transfer(task: Task) -> pd.DataFrame:
    """Train on all rows of two concepts, test on all rows of the third (no prompt is shared across concepts)."""
    rows = []
    for c in CONCEPTS:
        te = (task.meta.concept == c).values
        cond = task.meta.condition.values[te]
        for name, f in CLASSIFIERS.items():
            rows.append(dict(held_out=c, classifier=name,
                             **summary(task.y[te], f(task.X[~te], task.y[~te], task.X[te], task.X[~te]), cond)))
    return pd.DataFrame(rows).set_index(["held_out", "classifier"])


# ----------------------------------------------------------------------------- step 9

def mean_dependency(x: np.ndarray) -> float:
    """Average |P(Xi=1, Xj=1) - P(Xi=1) P(Xj=1)| over all pairs i < j (the researcher's Pintersect)."""
    d = np.abs(db.Pintersect(x))
    iu = np.triu_indices(d.shape[0], k=1)
    return float(d[iu].mean())


def dependency_by_group(task: Task, seed: int = SEED, by_concept: bool = True) -> pd.DataFrame:
    """Mean dependency between binarised dimensions per condition (none / own / other) on the training rows.
    Groups are subsampled to the same size, since the estimate of |dependency| grows with sampling noise at small n.
    `noise` row: two random halves of the same group, i.e. how much the number moves by chance alone."""
    Xtr, _ = task.train
    m = task.meta[~task.is_test].reset_index(drop=True)
    x = nb.mapping(Xtr, nb.medians(Xtr))
    rng = np.random.default_rng(seed)
    out = {}
    for c in (CONCEPTS if by_concept else [None]):
        sel = np.ones(len(m), bool) if c is None else (m.concept == c).values
        n = min(int((sel & (m.condition == k).values).sum()) for k in ("none", "own", "other"))
        row = {}
        for k in ("none", "own", "other"):
            idx = rng.choice(np.where(sel & (m.condition == k).values)[0], n, replace=False)
            row[k] = mean_dependency(x[idx])
        half = rng.permutation(np.where(sel & (m.condition == "none").values)[0])[: 2 * (n // 2)]
        row["noise (none vs none)"] = abs(mean_dependency(x[half[: n // 2]]) - mean_dependency(x[half[n // 2:]]))
        row["n per group"] = n
        out["all" if c is None else c] = row
    return pd.DataFrame(out).T
