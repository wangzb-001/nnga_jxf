import numpy as np
import os
from deap import base, tools, creator, algorithms
import random
import matplotlib.pyplot as plt
from src.tools import write_front
from src.GA_util2 import cxSimulatedBinaryBounded, Cross
from deap.benchmarks.tools import hypervolume
from src.problem import ZDT1, ZDT2, ZDT3, ZDT4

# 随机数种子，防止随机计算导致结果不可复现
random.seed(0)
# 定义问题
creator.create('MultiObjMin', base.Fitness, weights=(-1.0, -1.0))  # 两个目标，都求最小值
creator.create('Individual', list, fitness=creator.MultiObjMin)  # 创建individual类


def genInd(low, up, Ndim):
    pops = [random.uniform(low[0], up[0]) for _ in range(Ndim)]
    # return [random.uniform(low[0], up[0]), random.uniform(low[1], up[1])]  # 实数编码，一个个体里两个基因
    return pops


test_f = ZDT4
test_func = test_f.ZDT4
test_func_name = test_f.name
Ndim = test_f.dimention  # 变量维度

low = [-5] * Ndim  # 两个变量的下界
up = [5] * Ndim  # 两个变量的上界
low[0]=0
up[0]=1

N_POP = 200  # 种群内个体数量，参数过小，搜索速度过慢
NGEN = 500  # 迭代步数，参数过小，在收敛之前就结束搜索
CXPB = 0.8  # 交叉概率，参数过小，族群不能有效更新
MUTPB = 0.1  # 突变概率，参数过小，容易陷入局部最优
# 测试的交叉算子
# mate_name = "NSGA-II"
mate_name = "NSGA-II-NN"
name = f"{mate_name}-{NGEN}"
mate_s = {
    "NSGA-II": cxSimulatedBinaryBounded,
    "NSGA-II-NN": Cross,
}
toolbox = base.Toolbox()
toolbox.register('genInd', genInd, low, up, Ndim)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.genInd)  # 注册个体生成工具

# print(toolbox.individual())#打印一个个体检查


from src.nn.cross_ga_nn import CrossGaModel

CrossGaModel.init(dim=Ndim)

toolbox.register('evaluate', test_func)  # 注册评价函数

toolbox.register('population', tools.initRepeat, list, toolbox.individual)  # 注册种群生成工具
pop = toolbox.population(n=N_POP)  # 建立种群pop
# for ind in pop:#打印一个种群检查
#   print(ind)

toolbox.register('selectGen1', tools.selTournament, tournsize=2)
toolbox.register('select', tools.emo.selTournamentDCD)  # 该函数是binary tournament，不需要tournsize

# 交叉算子
toolbox.register('mate', mate_s[mate_name], eta=20.0, low=low, up=up)  # 交叉与变异方式选取较为讲究，随意换成其他方式效果不佳
toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=low, up=up, indpb=1.0 / Ndim)

# 开始迭代
# 第一代与第二代之后的代数操作不同
fitnesses = toolbox.map(toolbox.evaluate, pop)  # 为初始种群进行评价，得到适应度
for ind, fitness in zip(pop, fitnesses):
    ind.fitness.values = fitness  # 这里先有一个适应度才能进行快速非支配排序

fronts = tools.emo.sortNondominated(pop, k=N_POP, first_front_only=False)  # 快速非支配排序，得到不同前沿的pareto层集合fronts
# print(fronts)#打印一个pareto层集合fronts检查，每层前沿都是一个列表

for idx, front in enumerate(fronts):  # 使用枚举循环得到各层的标号与pareto解
    # print(idx,front)#打印检查前沿标号与每层的pareto解
    for ind in front:
        ind.fitness.values = (idx + 1),  # 将个体的适应度设定为pareto解的前沿次序

offspring = toolbox.selectGen1(pop, N_POP)  # 进行选择
offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)  # 只做一次交叉与变异操作

# 从第二代开始循环
for gen in range(1, NGEN):
    combinedPop = pop + offspring  # 将子代与父代结合成一个大种群

    fitnesses = toolbox.map(toolbox.evaluate, combinedPop)  # 对该大种群进行适应度计算
    for ind, fitness in zip(combinedPop, fitnesses):
        ind.fitness.values = fitness

    fronts = tools.emo.sortNondominated(combinedPop, k=N_POP, first_front_only=False)  # 对该大种群进行快速非支配排序

    for front in fronts:
        tools.emo.assignCrowdingDist(front)  # 计算拥挤距离
    pop = []
    for front in fronts:
        pop += front
    pop = toolbox.clone(pop)
    pop = tools.selNSGA2(pop, k=N_POP, nd='standard')  # 基于拥挤度实现精英保存策略

    offspring = toolbox.select(pop, N_POP)  # 选择
    offspring = toolbox.clone(offspring)
    offspring = algorithms.varAnd(offspring, toolbox, CXPB, MUTPB)  # 交叉变异

bestInd = tools.selBest(pop, 1)[0]  # 选择出种群中最优个体
bestFit = bestInd.fitness.values
print('best solution:', bestInd)
print('best fitness:', bestFit)

front = tools.emo.sortNondominated(pop, len(pop))[0]  # 返回的不同前沿的pareto层集合fronts中第一个front为当前最优解集
print(f"len of front:{len(front)}")

# 2目标，参考点
ref = [5, 5]
hypervolume_v = hypervolume(pop, ref=ref)
print(f"{name} : {hypervolume_v}")

write_front(front, hypervolume_v, f'{os.path.dirname(__file__)}/output/{test_func_name}/', f"{name}.txt")
# # 图形化显示
for ind in front:
    plt.plot(ind.fitness.values[0], ind.fitness.values[1], 'r.', ms=2)
plt.xlabel('f1')
plt.ylabel('f2')
plt.title(f"{test_func_name}: {name}")
plt.tight_layout()
plt.show()
