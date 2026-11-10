"""Naive Bayes on binarised activations: the researcher's code from notebooks/bayes.ipynb (cells 3 and 5),
copied verbatim so other notebooks can import it. The only change: `mapping` takes the thresholds as an argument
instead of reading the notebook's global `thresholds`."""
import numpy as np


def medians(X_train):
    return np.median(X_train, axis=0)


def mapping(X, thresholds):
    return (X > thresholds).astype(np.float32)


# we use logs so the scale of the numbers we are working with is reasonable
def logbernoulli(x, Px):
    return x @ np.log(Px) + (1 - x) @ np.log(1 - Px)

def fit(X, y):
    PH = y.mean()
    # adding 1 and 2 to prevent values of exactly 0 or 1 with log
    PxH  = (X[y == 1].sum(axis=0) + 1) / ((y == 1).sum() + 2)
    PxHc = (X[y == 0].sum(axis=0) + 1) / ((y == 0).sum() + 2)
    return PH, PxH, PxHc

def calcphx(x, model):
    PH, PxH, PxHc = model
    return 1 / (1 + np.exp((logbernoulli(x, PxHc) + np.log(1 - PH)) - (logbernoulli(x, PxH) + np.log(PH))))
