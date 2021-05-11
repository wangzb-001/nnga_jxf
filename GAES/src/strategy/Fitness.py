import numpy as np


def Fitness(test_func, pop, subject_id):
    Z = test_func.Func(pop, subject_id)
    f = np.arange(len(Z), 0, -1)[np.newaxis, :]
    a = np.argsort(Z)
    Z[a] = f
    fitness = Z
    return fitness
