#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/13 下午4:16
# @File    : utils.py
import pickle as pkl
import pandas as pd
import jieba


def save_pkl(obj, path):
    with open(path, mode='wb') as wf:
        pkl.dump(obj, wf)


def get_pkl(path):
    with open(path, mode='rb') as rf:
        return pkl.load(rf)


def load_stop_words(path='../data/HIT_stop_words.txt'):
    with open(path, encoding='utf-8', mode='r') as rf:
        lines = rf.readlines()
        lines = [line.strip() for line in lines]
        return lines


def cut(x, stop_words):
    xs = jieba.lcut(x)
    xs = [w for w in xs if w not in stop_words]
    return ' '.join(xs)


def load_data(path='../../data/Sentiment Classification with Deep Learning/test_label_cn_txt.csv', stop_words=[]):
    pd_data = pd.read_csv(path)
    pd_data['text'] = pd_data['text'].apply(lambda x: cut(x, stop_words))
    texts = pd_data.text.tolist()
    labels = pd_data.label.tolist()
    return texts, labels


def Cnk(n, k):
    s = 1
    for d in range(n, n - k, -1):
        s *= d
    for d in range(k, 0, -1):
        s = int(s / d)
    return s


if __name__ == '__main__':
    # load_stop_words()
    print(Cnk(10, 2))
