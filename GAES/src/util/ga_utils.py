#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/20 4:53 下午
# @File    : ga_utils.py
import os
from config import data_csv, data_dir

vocab_path = os.path.join(data_dir, 'vocab.txt')


def load_vocab():
    with open(vocab_path, mode='r', encoding='utf-8') as rf:
        vocabs = rf.readlines()
        words = [v.split('\t')[0] for v in vocabs]
        return words


words = load_vocab()
word2idx = {
    w: i for i, w in enumerate(words)
}
