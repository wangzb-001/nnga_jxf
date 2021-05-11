import numpy as np


def Sl_pop_by_roulette(fitness, pop):
    probility = fitness / fitness.sum()
    pop_size = len(pop)
    idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True,
                           p=probility)
    new_pop = []
    for i in idx:
        new_pop.append(pop[i])
    return new_pop
