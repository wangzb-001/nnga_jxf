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
    plt.title(f"{test_func_name}: {name}", fontsize=10)
    fronts = read_data(f"output/{test_func_name}/{name}.txt")
    for p in fronts:
        plt.plot(p[0], p[1], 'r.', ms=2)


plt.tick_params(labelsize=6)

fn = "ZDT4"
# plot(fn, name="NSGA-II-50", subplot=221)
# plot(fn, name="NSGA-II-100", subplot=222)
# plot(fn, name="NSGA-II-NN-50", subplot=223)
# plot(fn, name="NSGA-II-NN-100", subplot=224)

plot(fn, name="NSGA-II-150", subplot=221)
plot(fn, name="NSGA-II-200", subplot=222)
plot(fn, name="NSGA-II-NN-150", subplot=223)
plot(fn, name="NSGA-II-NN-200", subplot=224)



plt.tight_layout()
plt.show()
