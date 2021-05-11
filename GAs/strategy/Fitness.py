import numpy as np


def Fitness0(self, pop):
    Z = self.test_func.Func(pop)
    f = np.arange(self.test_func.pop_size, 0, -1)[np.newaxis, :]
    a = np.argsort(Z)
    Z[a] = f
    fitness = Z
    return fitness
