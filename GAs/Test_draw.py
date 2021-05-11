# !/usr/bin/python3
# 名称：遗传算法
# 时间: 2018/12/14 15:28
# 作者:jiangxinfa
# 邮件:425776024@qq.com
import numpy as np
import problems.Sample as Sample
from src.GeneticAlgorithm import *
from logers import LOGS
from src.nn.cross_ga_nn import CrossGaModel
import torch

seed = 0

acc_rate = 0.001
np.random.seed(seed)
torch.manual_seed(seed)
# 2：31
# 4：49
# 6：46
# 8：46
valuable_num, Bound, pop_size, iterator_num, golbal_fv, accuracy = 3, [-5, 10], 100, 10000, 0, acc_rate
rosenbrock = Rosenbrock.Rosenbrock(valuable_num, Bound, pop_size, iterator_num, golbal_fv, accuracy)
Funcs = []
Funcs.extend([rosenbrock])


# 2:1861
# 4:1967
# 6:
def F(fi, funci, ti, evo, cro, mu):
    namei = str(funci)
    mean_value = 0
    success_rate = 0
    mean_iterator = 0
    best = 100000
    worst = -100000
    mz_Arr = np.zeros(ti)
    Info = np.zeros(5)
    for i in range(ti):
        LOGS.init(f'logs/{fi}_log_{i}_{namei}.log')
        LOGS.log.debug(funci.__dict__)
        LOGS.log.debug('{}:{}--------------------------------', namei, i)
        ga = GeneticAlgorithm(cross_rate, mutation_rate, s_Create.Create1, funci, evo,
                              s_Fitness.Fitness0,
                              cro, mu, s_Select.Sl_pop_by_roulette, s_EO.EO0)
        mz, succ, it = ga.run(i)
        if mz < best:
            best = mz
        if mz > worst:
            worst = mz
        mz_Arr[i] = mz
        mean_value += mz
        mean_iterator += it
        if succ == 1:
            success_rate += 1
        LOGS.log.debug(funci.__dict__)
    LOGS.log.debug('------- {} end ------', namei)
    Info[0], Info[1], Info[2], Info[3], Info[4] = best, worst, mean_value / ti, success_rate / ti, mean_iterator / ti
    # np.savetxt('data/v/%s_%s.csv' % (namei, ti), np.array(mz_Arr))
    # np.savetxt('data/i/%s_info.csv' % (namei), Info)
    LOGS.log.debug(
        '{} : mean:{},success_rate:{},mean iterator:{}', namei, mean_value / ti, success_rate / ti, mean_iterator / ti)


def draw(func):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    X = []
    Y = []
    Data = []
    cld_x = Bound[0] + np.random.rand(valuable_num, 100) * (Bound[1] - Bound[0])
    for i in range(100):
        for j in range(100):
            x1, x2 = cld_x[:, i], cld_x[:, j]

            x3 = CrossGaModel.predict_one(x1, x2, func)
            x3 = x3.squeeze().detach().numpy()
            if x3.size == 1 and len(x3.shape) == 0:
                x3 = x3[np.newaxis, np.newaxis]
            r = np.random.rand()
            new_pop1 = np.multiply(x3, r * np.add(x1, x2))
            new_pop1[new_pop1 > Bound[1]] = Bound[1]
            new_pop1[new_pop1 < Bound[0]] = Bound[0]
            X.append([x1, x2, new_pop1])

            i1f = func(x1)
            i2f = func(x2)
            i3f = func(new_pop1)
            Y.append([i1f, i2f, i3f])
            Data.append([i, j, i3f.tolist()[0]])
        # print(i3f)

    data = np.array(Data)
    x = data[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y = data[:, 1]  # [ 1  4  7 10 13 16 19 22]
    z = data[:, 2]  # [ 2  5  8 11 14 17 20 23]
    LOGS.log.debug(f'np.sum(z): {np.sum(z)}')
    LOGS.log.debug(f'np.min(z): {np.min(z)}')
    LOGS.log.debug(f'np.max(z): {np.max(z)}')
    LOGS.log.debug(f'np.means(z): {np.mean(z)}')
    # 绘制散点图
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(x, y, z)
    #
    # # 添加坐标轴(顺序是Z, Y, X)
    # ax.set_zlabel('Z', fontdict={'size': 3, 'color': 'red'})
    # ax.set_ylabel('Y', fontdict={'size': 3, 'color': 'red'})
    # ax.set_xlabel('X', fontdict={'size': 3, 'color': 'red'})
    # plt.show()


if __name__ == '__main__':
    times = 1
    sEvolution = [s_Evolution.Evolution0]
    sCrossover = [s_Crossover.Crossover3]
    sMutate = [s_Mutate.Mutate0]
    cross_rate, mutation_rate = 0.5, 0.05
    i = 0
    CrossGaModel.init(dim=Funcs[i].valuable_num, test_func=Funcs[i])
    F(i, Funcs[i], times, sEvolution[0], sCrossover[0], sMutate[0])
    draw(Funcs[i].Func)

    # for i in range(1, 2):
    #     F(i, Funcs[i], times, sEvolution[0], sCrossover[0], sMutate[0])
