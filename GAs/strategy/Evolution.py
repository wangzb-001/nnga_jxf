import numpy as np
import strategy.Utils as utils
from logers import LOGS


def Evolution0(self):
    globle_min_X = 0
    self.pop = self.Create(self.test_func.Bound, self.test_func.valuable_num, self.test_func.pop_size)
    self.fitness = self.Fitness(self, self.pop)
    globle_min_z = float('Inf')
    gnenration_min_Z = np.array([])
    for gen in range(1, self.test_func.iterator_num):
        self.gen = gen
        # ============计算新的最优==============
        max_fitness_index = np.argmax(self.fitness)
        max_fitness_child = self.pop[:, max_fitness_index]
        max_fitness_value = self.test_func.Func(max_fitness_child)
        min_fitness_index = np.argmin(self.fitness)
        # self.same_size = utils.size(self, max_fitness_child)

        if max_fitness_value < globle_min_z:
            globle_min_z = max_fitness_value
            globle_min_X = max_fitness_child
            # LOGS.log.debug('th :{},fv:{},c:{},m:{}',
            #                gen, globle_min_z, self.cross_rate, self.mutation_rate)
            # LOGS.log.debug("x : {}", globle_min_X[:3])

            # self.rt -= self.test_func.accuracy
            # if self.rt <= self.test_func.accuracy:
            #     self.rt = self.test_func.accuracy
        # else:
        #     self.rt += self.test_func.accuracy
        #     if self.rt > 1:
        #         self.rt = 1
        # self.rt=1
        if max_fitness_value > globle_min_z:
            self.pop[:, min_fitness_index][:] = globle_min_X[:]
            # LOGS.log.debug('th :{},fv:{},c:{},m:{};rt:{};',
            #                gen, globle_min_z, self.cross_rate, self.mutation_rate, self.rt)
            if np.isnan(globle_min_X).any():
                print(1)
                break
            # LOGS.log.debug("x : {}", globle_min_X[:3])
        gnenration_min_Z = np.append(gnenration_min_Z, globle_min_z)

        for i in range(self.test_func.pop_size):
            j = np.random.randint(0, self.test_func.pop_size, size=1)[0]
            if i == j: continue
            p1 = self.pop[:, i]
            p2 = self.pop[:, j]
            self.Crossover(self, p1, p2)
            self.Mutate(self, p1, gen)
        self.fitness = self.Fitness(self, self.pop)
        self.pop = self.Select(self, self.pop)
        self.fitness = self.Fitness(self, self.pop)

        # ============计算新的最优==============
        max_fitness_index = np.argmax(self.fitness)
        max_fitness_child = self.pop[:, max_fitness_index]
        max_fitness_value = self.test_func.Func(max_fitness_child)

        if max_fitness_value < globle_min_z:
            globle_min_z = max_fitness_value
            globle_min_X = max_fitness_child
            # LOGS.log.debug('th :{},fv:{},c:{},m:{}',
            #                gen, globle_min_z, self.cross_rate, self.mutation_rate)
            # LOGS.log.debug("max x : {} , min x {}", max(globle_min_X), min(globle_min_X))
        if max_fitness_value > globle_min_z:
            self.pop[:, min_fitness_index][:] = globle_min_X[:]
        # ============计算新的最优==============
        if abs(
                globle_min_z - self.test_func.golbal_fv) <= self.test_func.accuracy or globle_min_z < self.test_func.golbal_fv:
            max_fitness_index = np.argmax(self.fitness)
            max_fitness_child = self.pop[:, max_fitness_index]
            max_fitness_value = self.test_func.Func(max_fitness_child)
            # self.same_size = utils.size(self, max_fitness_child)

            if max_fitness_value <= globle_min_z:
                globle_min_z = max_fitness_value
                globle_min_X = max_fitness_child
            gnenration_min_Z = np.append(gnenration_min_Z, globle_min_z)
            break
    return globle_min_X, globle_min_z, gnenration_min_Z
