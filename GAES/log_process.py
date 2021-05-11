#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/2/18 12:24 下午
# @File    : log_process.py

log_file = 'log_train1/eslog_sample.log'

lines = open(log_file).readlines()
nums = 0
better = 0
for line in lines:
    if 'score' in line:
        score_idx = line.index('score')
        score_str = line[score_idx:]
        score_i = score_str.index('[')
        score_j = score_str.index(']')
        score = score_str[score_i + 1:score_j]
        scores = score.split(',')
        scores = [float(s) for s in scores]
        if scores[-1] > scores[0]:
            better += 1
        nums += 1
        print(scores[0], ',', scores[-1])
        if nums % 100 == 0 and nums > 0:
            print(better, nums, better / nums, '\n')
            print('*' * 30)
        #     nums = 0
        #     better = 0
print(better, nums, better / nums)
