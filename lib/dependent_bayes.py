"""Dependent Bayes on binarised activations: the researcher's code from notebooks/bayes.ipynb (cells 8 and 9),
copied verbatim so other notebooks can import it. Input x is the output of lib.naive_bayes.mapping."""
import numpy as np


def Pintersect(x):
    n, dim = x.shape
    pi = (x).mean(axis=0) # when P(Xi = 1)
    pij = (x.T @ x) / n # P(Xi = 1 and Xj = 1)
    return pij - np.outer(pi, pi)         


def sortcorrelation(x, y):
    correlations = np.abs(Pintersect(x[y == 1])) + np.abs(Pintersect(x[y == 0]))
    dim = correlations.shape[0]
    order = np.argsort(-correlations.flatten())
    used = np.zeros(dim, dtype=bool)
    pairs = []
    for idx in order:
        i, j = idx//dim, idx%dim
        if i < j and not used[i] and not used[j]:   
            pairs.append((i, j))
            used[i] = True
            used[j] = True
    return np.array(pairs, dtype=int)

def statsperclass(xc, pairs, unpaired):
    n = len(xc)

    pi = np.zeros((len(unpaired), 2))
    for k in range(len(unpaired)):
        i = unpaired[k]
        pi[k, 0] = ((xc[:, i]==0).sum()+1) / (n+2)
        pi[k, 1] = ((xc[:, i]==1).sum()+1) / (n+2)

    pij = np.zeros((len(pairs), 2, 2)) # extra dimension to fit both i and j
    a = xc[:, pairs[:, 0]]     
    b = xc[:, pairs[:, 1]]     
    pij = np.zeros((len(pairs), 2, 2))
    pij[:, 0, 0] = (((a==0)&(b==0)).sum(axis=0)+1) / (n+4)   
    pij[:, 0, 1] = (((a==0)&(b==1)).sum(axis=0)+1) / (n+4)   
    pij[:, 1, 0] = (((a==1)&(b==0)).sum(axis=0)+1) / (n+4)   
    pij[:, 1, 1] = (((a==1)&(b==1)).sum(axis=0)+1) / (n+4)
    return np.log(pi), np.log(pij)

def fit(x, y, pairs):
    x = x.astype(int)
    dim = x.shape[1]
    paired = np.zeros(dim, dtype=bool)
    for i, j in pairs:
        paired[i] = True
        paired[j] = True
    unpaired = np.where(paired == False)[0]
    model = {"pairs": pairs, "unpaired": unpaired}

    x1 = x[y == 1]
    logPH = np.log(len(x1) / len(x))
    logpi, logpij = statsperclass(x1, pairs, unpaired)
    model["H"] = (logPH, logpi, logpij)

    x0 = x[y == 0]
    logPHc = np.log(len(x0) / len(x))
    logpi, logpij = statsperclass(x0, pairs, unpaired)
    model["H^c"] = (logPHc, logpi, logpij)
    return model

def score(x, model, option):
    logPclass, logpi, logpij = model[option]
    unpaired = model["unpaired"]
    pairs = model["pairs"]
    total = np.zeros(len(x))
    total += logPclass
    for k in range(len(unpaired)):
        i = unpaired[k]
        total += logpi[k, x[:, i]]
    for p in range(len(pairs)):
        i, j = pairs[p]
        total += logpij[p, x[:, i], x[:, j]]
    return total   

def bayespredict(x, model):
    x = x.astype(int)
    return score(x, model, "H") - score(x, model, "H^c")

