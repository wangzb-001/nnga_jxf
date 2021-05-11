# !/usr/bin/python3
# 名称：遗传算法
# 时间: 2018/12/14 15:28
# 作者:jiangxinfa
# 邮件:425776024@qq.com
import numpy as np
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
import problems.H16 as H16
from src.GeneticAlgorithm import *
from logers import LOGS
from src.nn.cross_ga_nn import CrossGaModel
from src.nn.mutate_ga_nn import MutateGaModel
import torch

seed = 0

acc_rate = 0.001
np.random.seed(seed)
torch.manual_seed(seed)

ackley = Ackley.Ackley(50, [-32, 32], 30, 20000, 0, acc_rate)
griewank = Griewank.Griewank(50, [-600, 600], 30, 20000, 0, acc_rate)
# michalewicz = Michalewicz.Michalewicz(10, 10, [0.00, np.pi], 30, 20000, -9.66015, 0.0001)
michalewicz = Michalewicz.Michalewicz(10, 5, [0.00, np.pi], 30, 50000, -4.687658, acc_rate)
# michalewicz = Michalewicz.Michalewicz(10, 2, [0.00, np.pi], 30, 20000, -1.801, 0.0001)
quartic = Quartic.Quartic(50, [-100, 100], 30, 30000, 0, acc_rate)
rastrigin = Rastrigin.Rastrigin(50, [-5.12, 5.12], 30, 50000, 0, acc_rate)
rosenbrock = Rosenbrock.Rosenbrock(3, [-5, 10], 30, 100000, 0, acc_rate)
schwefel = Schwefel.Schwefel(3, [-500, 500], 30, 20000, 0, acc_rate)
sphere = Sphere.Sphere(50, [-100, 100], 30, 20000, 0, acc_rate)
step = Step.Step(50, [-100, 100], 30, 30000, 0, acc_rate)

h1 = H1.H1(100, [-0.5, 0.5], 30, 10000, 0, acc_rate)
# h5 = H5.H5(5, [-1, 2], 30, 20000, -1.85 * 5, acc_rate)
h6 = H6.H6(100, [3, 13], 30, 20000, -1.21598 * 100, acc_rate)
h9 = H9.H9(100, [-5.12, 5.12], 30, 20000, 0, acc_rate)
h13 = H13.H13(100, [-100, 100], 30, 100000, 0, acc_rate)
h14 = H14.H14(100, [-1.28, 1.28], 30, 100000, 0, acc_rate)
h16 = H16.H16(100, [-10, 10], 30, 100000, 0, acc_rate)

Funcs = []
Funcs.extend([ackley, griewank, michalewicz, quartic, rastrigin, rosenbrock, schwefel, sphere, step])
Funcs.append(h1)
# Funcs.append(h5)
Funcs.append(h6)
Funcs.append(h9)
Funcs.append(h13)
Funcs.append(h14)
Funcs.append(h16)


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


if __name__ == '__main__':
    times = 1
    sEvolution = [s_Evolution.Evolution0]
    sCrossover = [s_Crossover.Crossover3]
    sMutate = [s_Mutate.Mutate1]
    cross_rate, mutation_rate = 0.5, 0.05
    il = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    i = 1
    CrossGaModel.init(dim=Funcs[i].valuable_num, test_func=Funcs[i])
    # MutateGaModel.init(dim=Funcs[i].valuable_num, test_func=Funcs[i])
    F(i, Funcs[i], times, sEvolution[0], sCrossover[0], sMutate[0])
