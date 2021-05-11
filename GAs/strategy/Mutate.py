import numpy as np
from strategy.Utils import func_levy_v, func_levy, Levy
from src.nn.mutate_ga_nn import MutateGaModel
from logers import LOGS


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


def Mutate1(self, pop, gen):
    Bound = self.test_func.Bound
    d = (Bound[1] - Bound[0]) / 100
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


def Mutate1_new(self, pop1, gen):
    Bound = self.test_func.Bound
    if pop1.size == 1:
        pop1 = pop1[:, np.newaxis]
    if np.random.rand() < self.mutation_rate:
        p1_f = self.test_func.Func(pop1)
        x1 = MutateGaModel.predict_one(pop1, func=self.test_func.Func)
        x1 = x1.squeeze().detach().numpy()
        if x1.size == 1 and len(x1.shape) == 0:
            x1 = x1[np.newaxis, np.newaxis]
        new_pop1 = x1
        new_pop1[new_pop1 > Bound[1]] = Bound[1]
        new_pop1[new_pop1 < Bound[0]] = Bound[0]
        new_pop1_f = self.test_func.Func(new_pop1)
        log = False
        x = None
        if log: LOGS.log.debug(f"进行了一次预测，{pop1[:3]},x:{new_pop1[:3]},预测结果：{new_pop1_f}")
        # pop1最优
        if p1_f < new_pop1_f:
            if log: LOGS.log.debug(f"更优下一代：{new_pop1[:3]} ->{pop1[:3]}...，fv :{p1_f}，开始进行一次训练")
            MutateGaModel.train_one(new_pop1, pop1, func=self.test_func.Func)

        # new_pop1最优
        if new_pop1_f < p1_f:
            if log: LOGS.log.debug(f"更优下一代：{pop1[:3]}->{new_pop1[:3]}...，fv :{new_pop1_f}，开始进行一次训练")
            MutateGaModel.train_one(pop1, new_pop1, func=self.test_func.Func)
            x = new_pop1
        if x is None:
            tp_ra = self.mutation_rate
            self.mutation_rate = 1
            pop1 = Mutate1(self, pop1, gen)
            self.mutation_rate = tp_ra
        else:
            # 如果随机的更优，交叉的后代是随机的那个新个体
            pop1[:] = x[:]

    return pop1


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
