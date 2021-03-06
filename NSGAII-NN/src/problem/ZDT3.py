'''
求解问题部分
'''

import numpy as np

# 函数的维度（目标维度不一致的自行编写目标函数）
dimention = 30
# 目标函数个数
Func_num = 2
bound = [0, 1]
name = "ZDT3"


def Func(X):
    f1 = F1(X)
    gx = g(X)
    f2 = F2(gx, X)
    return [f1, f2]


def F1(X):
    return X[0]


def F2(gx, X):
    x1 = X[0]
    f2 = gx * (1 - np.sqrt(x1 / gx) - (x1 / gx) * np.sin(10 * np.pi * x1))
    return f2


def g(X):
    g = 1 + 9 * (np.sum(X[1:], axis=0) / (len(X) - 1))
    return g


def ZDT3(ind):
    n = len(ind)
    f1 = ind[0]
    g = 1 + 9 * np.sum(ind[1:]) / (n - 1)
    f2 = g * (1 - np.sqrt(ind[0] / g) - ind[0] / g * np.sin(10 * np.pi * ind[0]))
    return f1, f2