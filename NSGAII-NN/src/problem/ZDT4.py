'''
求解问题部分
'''

import numpy as np

name = "ZDT4"
dimention = 10
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
    x = X[0]
    f2 = gx * (1 - np.sqrt(x / gx))
    return f2


def g(X):
    g = 1 + 10 * (X.shape[0] - 1) + (np.sum(X[1:] ** 2 - 10 * np.cos(4 * np.pi * X[1:]), axis=0))
    return g


def ZDT4(ind):
    n = len(ind)
    f1 = ind[0]
    np_ind = np.array(ind[1:])
    g = 1 + 10 * (n - 1) + (np.sum(np_ind ** 2 - 10 * np.cos(4 * np.pi * np_ind), axis=0))
    f2 = g * (1 - np.sqrt(f1 / g))

    return f1, f2
