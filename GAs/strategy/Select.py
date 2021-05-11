import numpy as np


def Sl_pop_by_roulette(self, pop):
    probility = self.fitness / self.fitness.sum()
    idx = np.random.choice(np.arange(self.test_func.pop_size), size=self.test_func.pop_size, replace=True,
                           p=probility)
    return pop[:, idx]
