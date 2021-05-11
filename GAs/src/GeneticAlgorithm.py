import time
import numpy as np
import strategy.Utils as utils
import problems.Ackley as Ackley
import problems.Griewank as Griewank
import problems.Michalewicz as Michalewicz
import problems.Quartic as Quartic
import problems.Rastrigin as Rastrigin
import problems.Rosenbrock as Rosenbrock
import problems.Schwefel as Schwefel
import problems.Schwefel_2p22 as Schwefel_2p22
import problems.Sphere as Sphere
import problems.Step as Step
import problems.H1 as H1
import problems.H5 as H5
import problems.H6 as H6
import problems.H9 as H9
import problems.H13 as H13
import problems.H14 as H14

import strategy.Create as s_Create
import strategy.Cross as s_Crossover
import strategy.EO as s_EO
import strategy.Evolution as s_Evolution
import strategy.Fitness as s_Fitness
import strategy.Mutate as s_Mutate
import strategy.Select as s_Select
from logers import LOGS


class GeneticAlgorithm():
    cross_rate = 0.00
    mutation_rate = 0.00
    fitness = []  # 记录种群适应度
    pop = []  # 记录种群适应度
    rt = 1

    Create = None
    test_func = None
    Evolution = None
    EO = None
    Fitness = None
    Crossover = None
    Mutate = None
    Select = None

    def __init__(self, cross_rate, mutation_rate, Create, test_func, Evolution, Fitness, Crossover, Mutate, Select, EO):
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.test_func = test_func
        self.rt = 1
        self.Create = Create
        self.EO = EO
        self.Evolution = Evolution
        self.Fitness = Fitness
        self.Crossover = Crossover
        self.Mutate = Mutate
        self.Select = Select

    def run(self, i):
        t = time.time()
        mx, mz, gz = self.Evolution(self)
        if np.abs(mz - self.test_func.golbal_fv) <= self.test_func.accuracy or mz < self.test_func.golbal_fv:
            success = 1
        else:
            success = 0
        dt = time.time() - t
        LOGS.log.debug('th {} ，fv：{},max x:{}', i, mz, max(mx))
        LOGS.log.debug('th {} ，fv：{},min x:{}', i, mz, min(mx))
        LOGS.log.debug('x:{},time:{}', mx, dt)
        tx = utils.pow_exe(self.test_func.Bound, mx)
        ty = self.test_func.Func(mx)
        # print(ty, tx)
        return mz, success, gz.shape[0]
