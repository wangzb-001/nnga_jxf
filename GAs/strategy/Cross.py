import numpy as np
from strategy.Utils import func_levy_v, func_levy, Levy
from src.nn.cross_ga_nn import CrossGaModel
from logers import LOGS
import torch.nn.functional as f


def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))


def Crossover0(self, pop1, pop2):
    Bound = self.test_func.Bound
    if np.random.rand() < self.cross_rate:
        p1_f = self.test_func.Func(pop1)
        p2_f = self.test_func.Func(pop2)
        r = np.random.rand()
        if p1_f < p2_f:
            x = r * (pop1 - pop2) + pop1
        else:
            x = r * (pop2 - pop1) + pop2
        pop1[:] = x[:]
        pop1[pop1 > Bound[1]] = Bound[1]
        pop1[pop1 < Bound[0]] = Bound[0]
    return pop1, pop2


def Crossover1(self, pop1, pop2):
    Bound = self.test_func.Bound
    d = (Bound[1] - Bound[0]) / 1000
    if np.random.rand() < self.cross_rate:
        p1_f = self.test_func.Func(pop1)
        p2_f = self.test_func.Func(pop2)
        r = np.random.rand()
        if p1_f < p2_f and abs(p1_f - p2_f) > self.test_func.accuracy / 10:
            x = r * (pop1 - pop2) + pop1
        elif p1_f > p2_f and abs(p1_f - p2_f) > self.test_func.accuracy / 10:
            x = r * (pop2 - pop1) + pop2
        else:
            # x = func_levy((pop1 + pop2) / 2, ala=d)
            x = func_levy(pop1, ala=d)
        pop1[:] = x[:]
        pop1[pop1 > Bound[1]] = Bound[1]
        pop1[pop1 < Bound[0]] = Bound[0]
    return pop1, pop2


def Crossover2(self, pop1, pop2):
    Bound = self.test_func.Bound
    d = (Bound[1] - Bound[0]) / 1000
    if np.random.rand() < self.cross_rate:
        r = np.random.rand()
        x = r * (pop1 - pop2) + pop1
        pop1[:] = x[:]
        pop1[pop1 > Bound[1]] = Bound[1]
        pop1[pop1 < Bound[0]] = Bound[0]
    return pop1, pop2


def Crossover3(self, pop1, pop2):
    Bound = self.test_func.Bound
    # d = (Bound[1] - Bound[0]) / 1000
    if pop1.size == 1:
        pop1 = pop1[:, np.newaxis]
        pop2 = pop2[:, np.newaxis]
    if np.random.rand() < self.cross_rate:
        p1_f = self.test_func.Func(pop1)
        p2_f = self.test_func.Func(pop2)
        x1 = CrossGaModel.predict_one(pop1, pop2, func=self.test_func.Func)

        x1 = x1.squeeze().detach().numpy()
        if x1.size == 1 and len(x1.shape) == 0:
            x1 = x1[np.newaxis, np.newaxis]
        new_pop1 = x1
        new_pop1[new_pop1 > Bound[1]] = Bound[1]
        new_pop1[new_pop1 < Bound[0]] = Bound[0]
        new_pop1_f = self.test_func.Func(new_pop1)
        log = False
        x = None
        if log: LOGS.log.debug(f"进行了一次预测，{pop1[:3], pop2[:3]},x:{new_pop1[:3]},预测结果：{new_pop1_f}")
        # pop1最优
        if p1_f < new_pop1_f and p1_f < p2_f:
            if log: LOGS.log.debug(f"更优下一代：{new_pop1[:3], pop2[:3]} ->{pop1[:3]}...，fv :{p1_f}，开始进行一次训练")
            CrossGaModel.train_one(new_pop1, pop2, pop1, func=self.test_func.Func)
            CrossGaModel.train_one(pop2, new_pop1, pop1, func=self.test_func.Func)

            CrossGaModel.train_one(new_pop1, new_pop1, pop1, func=self.test_func.Func)
            CrossGaModel.train_one(pop2, pop2, pop1, func=self.test_func.Func)
            # x = pop1

        # pop2最优
        if p2_f < new_pop1_f and p2_f < p1_f:
            if log: LOGS.log.debug(f"更优下一代：{new_pop1[:3], pop1[:3]}->{pop2[:3]}...，fv :{p2_f}，开始进行一次训练")
            CrossGaModel.train_one(new_pop1, pop1, pop2, func=self.test_func.Func)
            CrossGaModel.train_one(pop1, new_pop1, pop2, func=self.test_func.Func)
            CrossGaModel.train_one(new_pop1, new_pop1, pop2, func=self.test_func.Func)
            CrossGaModel.train_one(pop1, pop1, pop2, func=self.test_func.Func)
            # x = pop2
        # new_pop1最优
        if new_pop1_f < p1_f and new_pop1_f < p2_f:
            if log: LOGS.log.debug(f"更优下一代：{pop1[:3], pop2[:3]}->{new_pop1[:3]}...，fv :{new_pop1_f}，开始进行一次训练")
            CrossGaModel.train_one(pop1, pop2, new_pop1, func=self.test_func.Func)
            CrossGaModel.train_one(pop2, pop1, new_pop1, func=self.test_func.Func)
            CrossGaModel.train_one(pop1, pop1, new_pop1, func=self.test_func.Func)
            CrossGaModel.train_one(pop2, pop2, new_pop1, func=self.test_func.Func)
            x = new_pop1
        if x is None:
            pop1, pop2 = Crossover1(self, pop1, pop2)
            # pop1, pop2 = Crossover2(self, pop1, pop2)
        else:
            # 如果随机的更优，交叉的后代是随机的那个新个体
            pop1[:] = x[:]

    return pop1, pop2
