import time
import numpy as np
from src.logers import LOGS
import src.strategy.Create as s_Create
import src.strategy.Cross as s_Crossover
import src.strategy.Evolution as s_Evolution
import src.strategy.Fitness as s_Fitness
import src.strategy.Mutate as s_Mutate
import src.strategy.Select as s_Select
from src.ES_predict import search
from src.nn.cross_ga_nn import CrossGaModel
import torch
import torch.nn.functional as F


class GeneticAlgorithm:
    cross_rate = 0.00
    mutation_rate = 0.00
    fitness = []  # 记录种群适应度
    pop = []  # 记录种群适应度
    rt = 1

    test_func = None

    def __init__(self, cross_rate, mutation_rate, test_func):
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.test_func = test_func
        self.rt = 1

    def test(self, words, subject_id):
        if isinstance(subject_id, int):
            subject_id = str(subject_id)
        pop1 = s_Create.Create(words, size=1)
        if pop1 is None:
            return [0, 0]
        pop1 = pop1[0]
        pop2 = pop1.copy()
        one_idx = pop2.index(1)
        pop2[one_idx] = 0
        p1_f, p2_f = self.test_func.Func([pop1, pop2], subject_id).tolist()
        x1 = CrossGaModel.predict_one(pop1, pop2, p1_f, p2_f)
        new_pop = torch.sigmoid(x1).squeeze().detach().numpy()
        del x1
        new_pop = new_pop.tolist()
        sn = sum(new_pop)
        LOGS.log.debug(f'[sample]sumx:{sn}')
        s1 = ','.join(words)
        mw = self.test_func.pop_to_words(new_pop)
        s2 = ','.join(mw)
        scores = search([s1, s2], [subject_id] * 2)
        return scores

    def run(self, words, subject_id):
        if isinstance(subject_id, int):
            subject_id = str(subject_id)
        t = time.time()
        try:
            mx, mz, gz = s_Evolution.evolution(self, words, subject_id)
        except Exception as e:
            print(e)
            return None
        if np.abs(mz - self.test_func.golbal_fv) <= self.test_func.accuracy or mz < self.test_func.golbal_fv:
            success = 1
        else:
            success = 0
        dt = time.time() - t
        # LOGS.log.debug('th {} ，fv：{},min x :{} ,max x:{}', words, mz, min(mx), max(mx))
        # LOGS.log.debug('x:{},time:{}', mx, dt)

        s1 = ','.join(words)
        mw = self.test_func.pop_to_words(mx)
        s2 = ','.join(mw)
        LOGS.log.debug('[sample]\n')
        LOGS.log.debug(f'[sample]pri:{s1}')
        LOGS.log.debug(f'[sample]aug:{s2}')
        scores = search([s1, s2], [subject_id] * 2)
        LOGS.log.debug(f'[sample]score:{scores}')
        # LOGS.log.debug("sum x : {}", sum(mx))
        LOGS.log.debug('[sample]\n')
        # fv = [1000 / (ps + 0.001) for ps in scores]
