'''
求解问题部分
'''


import numpy as np

name="ZDT1"
dimention = 30
bound = [0, 1]


def Func(X):
    if X.shape[0] < 2:
        return -1
    f1 = F1(X)
    ag = g(X)
    f2 = F2(ag, X)
    return [f1, f2]


def F1(X):
    return X[0]


def F2(gx, X):
    x=X[0]
    f2 = gx * (1 - np.sqrt(x / gx))
    return f2


def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (X.shape[0] - 1))
    return g

def ZDT1(ind):
    n = len(ind)
    f1 = ind[0]
    g = 1 + 9 * np.sum(ind[1:]) / (n - 1)
    f2=g * (1 - np.sqrt(ind[0] / g))
    return f1, f2