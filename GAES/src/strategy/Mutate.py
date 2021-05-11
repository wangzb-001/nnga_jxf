import numpy as np
from src.strategy.Utils import func_levy_v, func_levy, Levy
from src.nn.cross_ga_nn import CrossGaModel


def Mutate0(self, pop, gen):
    Bound = self.test_func.Bound
    # 标准模拟二进制
    if np.random.rand() < self.mutation_rate:
        temp = np.copy(pop)
        for i in range(np.random.randint(1, self.test_func.valuable_num)):
            k = np.random.randint(0, self.test_func.valuable_num, size=1)[0]
            temp[k] = temp[k] + np.random.normal(0, self.rt)
        temp[temp > Bound[1]] = Bound[1]
        temp[temp < Bound[0]] = Bound[0]
        pop[:] = temp[:]
    return pop


def Mutate1(self, pop, subject_id):
    Bound = self.test_func.Bound
    d = (Bound[1] - Bound[0])
    # 部分萊维飞行
    if np.random.rand() < self.mutation_rate:
        temp = np.copy(pop)
        r_num = np.random.randint(0, self.test_func.valuable_num)
        if r_num == 0: r_num = 1
        for i in range(r_num):
            k = np.random.randint(0, self.test_func.valuable_num, size=1)[0]
            temp[k] = func_levy_v(temp[k], ala=d)
        temp[temp > Bound[1]] = Bound[1]
        temp[temp < Bound[0]] = Bound[0]
        pop[:] = temp[:]
    return pop


def Mutate1_new(self, pop, gen):
    Bound = self.test_func.Bound
    d = (Bound[1] - Bound[0]) / 100
    # 部分萊维飞行
    if np.random.rand() < self.mutation_rate:
        temp = np.copy(pop)
        for i in range(np.random.randint(1, self.test_func.valuable_num)):
            k = np.random.randint(0, self.test_func.valuable_num, size=1)[0]
            temp[k] = temp[k] + Levy.levy_flight(n=1, m=1)[0] * d
        temp[temp > Bound[1]] = Bound[1]
        temp[temp < Bound[0]] = Bound[0]
        pop[:] = temp[:]
    return pop


def Mutate2(self, pop, gen):
    Bound = self.test_func.Bound
    d = (Bound[1] - Bound[0]) / 10
    # 全部萊维飞行
    if np.random.rand() < self.mutation_rate:
        temp = np.copy(pop)
        temp = func_levy(temp, ala=d)
        temp[temp > Bound[1]] = Bound[1]
        temp[temp < Bound[0]] = Bound[0]
        pop[:] = temp[:]
    return pop


def Mutate3(self, pop, gen):
    Bound = self.test_func.Bound
    # 标准模拟二进制
    if np.random.rand() < self.mutation_rate:
        x1 = CrossGaModel.predict_one(pop, pop, func=self.test_func.Func)
        x1 = x1.squeeze().detach().numpy()
        x1[x1 > Bound[1]] = Bound[1]
        x1[x1 < Bound[0]] = Bound[0]
        pop[:] = x1[:]
    return pop
