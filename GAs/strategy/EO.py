import numpy as np


def EO0(self, pop, test_func, rt):
    Bound = self.test_func.Bound
    m = pop.shape[0]
    tt = np.diag(np.ones(m))
    temp = np.tile(pop, (m, 1)).T
    step_rd = np.random.normal(0, 0.1, (m, 1))
    tt *= step_rd
    temp += tt
    temp[temp > Bound[1]] = Bound[1]
    temp[temp < Bound[0]] = Bound[0]
    F = test_func.Func(temp)
    j = np.argmin(F)
    best = temp[:, j]
    if test_func.Func(best) < test_func.Func(pop):
        pop[:] = best[:]
    return pop
