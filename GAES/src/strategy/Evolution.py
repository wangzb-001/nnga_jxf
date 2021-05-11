import numpy as np
from src.logers import LOGS
import src.strategy.Create as s_Create
import src.strategy.Cross as s_Crossover
import src.strategy.Fitness as s_Fitness
import src.strategy.Mutate as s_Mutate
import src.strategy.Select as s_Select


def evolution(ga_self, words: list, subject_id):
    '''
    输入一条句子词组，对其进行编码进化，训练
    :param sentence:
    :param subject_id:
    :return:
    '''
    pops = s_Create.Create(words, size=6)
    if pops is None:
        return None
    fitness = s_Fitness.Fitness(ga_self.test_func, pops, subject_id)
    pop_size = len(pops)
    globle_min_z = float('Inf')
    gnenration_min_Z = np.array([])
    globle_min_X = []
    for gen in range(0, ga_self.test_func.iterator_num):
        # ============计算新的最优==============
        max_fitness_index = np.argmax(fitness)
        max_fitness_child = pops[max_fitness_index].copy()
        max_fitness_value = ga_self.test_func.Func([max_fitness_child], subject_id)[0]
        min_fitness_index = np.argmin(fitness)

        if max_fitness_value < globle_min_z:
            globle_min_z = max_fitness_value
            globle_min_X = max_fitness_child
            # LOGS.log.debug('th :{},fv:{},c:{},m:{}',
            #                gen, globle_min_z, ga_self.cross_rate, ga_self.mutation_rate)
            # LOGS.log.debug("sum x : {}", sum(globle_min_X))

        # if max_fitness_value > globle_min_z:
        #     pops[min_fitness_index][:] = globle_min_X[:]
        #     LOGS.log.debug('th :{},fv:{},c:{},m:{};rt:{};',
        #                    gen, globle_min_z, ga_self.cross_rate, ga_self.mutation_rate, ga_self.rt)
        #     if np.isnan(globle_min_X).any():
        #         print(1, globle_min_X)
        #     LOGS.log.debug("x : {}", globle_min_X[:3])
        gnenration_min_Z = np.append(gnenration_min_Z, globle_min_z)

        for i in range(pop_size):
            j = np.random.randint(0, pop_size, size=1)[0]
            if i == j: continue
            p1 = pops[i]
            p2 = pops[j]
            p1[:], p2[:] = s_Crossover.Crossover3(ga_self, p1, p2, subject_id)
            p1[:] = s_Mutate.Mutate1(ga_self, p1, subject_id)
        fitness = s_Fitness.Fitness(ga_self.test_func, pops, subject_id)
        pops = s_Select.Sl_pop_by_roulette(fitness, pops)

        # ============计算新的最优==============
        max_fitness_index = np.argmax(fitness)
        max_fitness_child = pops[max_fitness_index].copy()
        max_fitness_value = ga_self.test_func.Func([max_fitness_child], subject_id)[0]

        if max_fitness_value <= globle_min_z:
            globle_min_z = max_fitness_value
            globle_min_X = max_fitness_child.copy()
            # LOGS.log.debug('th :{},fv:{},c:{},m:{}',
            #                gen, globle_min_z, ga_self.cross_rate, ga_self.mutation_rate)
            # LOGS.log.debug("max x : {} , min x {}", max(globle_min_X), min(globle_min_X))
        if max_fitness_value > globle_min_z:
            pops[max_fitness_index] = globle_min_X.copy()
        # ============计算新的最优==============
        if abs(
                globle_min_z - ga_self.test_func.golbal_fv) <= ga_self.test_func.accuracy or globle_min_z < ga_self.test_func.golbal_fv:
            max_fitness_index = np.argmax(fitness)
            max_fitness_child = pops[max_fitness_index].copy()
            max_fitness_value = ga_self.test_func.Func([max_fitness_child], subject_id)[0]
            # self.same_size = utils.size(self, max_fitness_child)

            if max_fitness_value <= globle_min_z:
                globle_min_z = max_fitness_value
                globle_min_X = max_fitness_child.copy()
            gnenration_min_Z = np.append(gnenration_min_Z, globle_min_z)
            break
    return globle_min_X, globle_min_z, gnenration_min_Z
