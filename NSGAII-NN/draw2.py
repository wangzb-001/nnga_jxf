import numpy as np
import os
import matplotlib.pyplot as plt
from src.tools import write_front


def read_data(path):
    fronts = []
    with open(path, mode='r') as rf:
        _ = rf.readline()
        for line in rf.readlines():
            p = line.strip().split('\t')
            fronts.append([float(p[0]), float(p[1])])
    return fronts


def plot(test_func_name="ZDT1", name="NSGA-II-50", subplot=221):
    plt.subplot(subplot)
    plt.xlabel('f1', fontsize=10)
    plt.ylabel('f2', fontsize=10)
    plt.title(f"{test_func_name}'s Pareto-optimal front", fontsize=10)
    fronts = read_data(f"output/{test_func_name}/{name}.txt")
    for p in fronts:
        plt.plot(p[0], p[1], 'r.', ms=2)


plt.tick_params(labelsize=6)
plot("ZDT1", name="NSGA-II-NN-500", subplot=221)
plot("ZDT2", name="NSGA-II-NN-500", subplot=222)
plot("ZDT3", name="NSGA-II-NN-500", subplot=223)
plot("ZDT4", name="NSGA-II-NN-500", subplot=224)

plt.tight_layout()
plt.show()
