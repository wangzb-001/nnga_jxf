#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/15 3:50 下午
# @File    : tools.py
import os


def write_front(front, hypervolume_v, path, name):
    if not os.path.exists(path):
        os.mkdir(path)
    with open(os.path.join(path, name), mode='w+', encoding='utf-8') as wf:
        wf.write(f"hypervolume : {hypervolume_v}\n")
        for ind in front:
            f1, f2 = ind.fitness.values[0], ind.fitness.values[1]
            wf.write(f"{f1}\t{f2}\n")
