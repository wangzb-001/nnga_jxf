import random
import warnings

from collections import Sequence
from itertools import repeat


def cxSimulatedBinaryBounded(ind1, ind2, eta, low, up):
    size = min(len(ind1), len(ind2))
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of the shorter individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of the shorter individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):
        if random.random() <= 0.5:
            # This epsilon should probably be changed for 0 since
            # floating point arithmetic in Python is safer
            if abs(ind1[i] - ind2[i]) > 1e-14:
                x1 = min(ind1[i], ind2[i])
                x2 = max(ind1[i], ind2[i])
                rand = random.random()

                beta = 1.0 + (2.0 * (x1 - xl) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

                c1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1))

                beta = 1.0 + (2.0 * (xu - x2) / (x2 - x1))
                alpha = 2.0 - beta ** -(eta + 1)
                if rand <= 1.0 / alpha:
                    beta_q = (rand * alpha) ** (1.0 / (eta + 1))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                c2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1))

                c1 = min(max(c1, xl), xu)
                c2 = min(max(c2, xl), xu)

                if random.random() <= 0.5:
                    ind1[i] = c2
                    ind2[i] = c1
                else:
                    ind1[i] = c1
                    ind2[i] = c2

    return ind1, ind2


import numpy as np
import random
import torch
import src.problem.SCH  as SCH
import src.problem.ZDT1 as ZDT1
import src.problem.ZDT2 as ZDT2
import src.problem.ZDT3 as ZDT3
import src.problem.ZDT4 as ZDT4
import src.problem.DTLZ1 as DTLZ1
from src.nn.cross_ga_nn import CrossGaModel

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

'''
遗传算法的一些辅助操作
'''
# 交叉率
c_rate = 0.8
# 突变率
m_rate = 0.1
# 测试函数/基于你自己的应用修改
test_fun = ZDT3
# 个体/class Individual 对象
idv = None


def is_dominate(a_F_value, b_F_value):
    # a是否支配b
    i = 0
    for av, bv in zip(a_F_value, b_F_value):
        if av < bv:
            i = i + 1
        if av > bv:
            return False
    if i != 0:
        return True
    return False


def Cross(ind1, ind2, eta, low, up):
    ind1 = cross2(ind1, ind2, eta, low, up)
    ind2 = cross2(ind2, ind1, eta, low, up)
    return ind1, ind2


def cross2(ind1, ind2, eta, low, up):
    Bound = [0, 1]
    Func = test_fun.Func
    x = None
    if np.random.rand() < c_rate:

        pop1 = list(ind1)
        pop2 = list(ind2)
        x1 = CrossGaModel.predict_one(pop1, pop2, func=Func)

        x1 = x1.squeeze().detach().numpy()
        if x1.size == 1 and len(x1.shape) == 0:
            x1 = x1[np.newaxis, np.newaxis]
        x1[x1 > Bound[1]] = Bound[1]
        x1[x1 < Bound[0]] = Bound[0]
        new_pop1 = x1.tolist()

        p1_f = Func(pop1)
        p2_f = Func(pop2)
        new_pop1_f = Func(new_pop1)

        # pop1最优
        if is_dominate(p1_f, new_pop1_f) and is_dominate(p1_f, p2_f):
            CrossGaModel.train_one(new_pop1, pop2, pop1, func=Func)
            CrossGaModel.train_one(pop2, new_pop1, pop1, func=Func)

            CrossGaModel.train_one(new_pop1, new_pop1, pop1, func=Func)
            CrossGaModel.train_one(pop2, pop2, pop1, func=Func)
            # x = pop1

        # pop2最优
        if is_dominate(p2_f, new_pop1_f) and is_dominate(p2_f, p1_f):
            CrossGaModel.train_one(new_pop1, pop1, pop2, func=Func)
            CrossGaModel.train_one(pop1, new_pop1, pop2, func=Func)
            CrossGaModel.train_one(new_pop1, new_pop1, pop2, func=Func)
            CrossGaModel.train_one(pop1, pop1, pop2, func=Func)
            # x = pop2
        # new_pop1最优
        if is_dominate(new_pop1_f, p1_f) and is_dominate(new_pop1_f, p2_f):
            CrossGaModel.train_one(pop1, pop2, new_pop1, func=Func)
            CrossGaModel.train_one(pop2, pop1, new_pop1, func=Func)
            CrossGaModel.train_one(pop1, pop1, new_pop1, func=Func)
            CrossGaModel.train_one(pop2, pop2, new_pop1, func=Func)
            x = new_pop1
    if x is None:
        ind1, ind2 = cxSimulatedBinaryBounded(ind1, ind2, eta, low, up)
    else:
        ind1[0:] = x[0:]
    return ind1
