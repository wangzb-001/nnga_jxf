'''
求解问题部分
'''

import numpy as np

dimention = 30
bound = [-1000, 1000]


def Func(X):
    f1 = F1(X)
    f2 = F2(X)
    return [f1, f2]


def F1(X):
    return X[0] ** 2


def F2(X):
    return (X[0] - 2) ** 2
