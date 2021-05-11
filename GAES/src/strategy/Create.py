import numpy as np
import os
from src.util.ga_utils import word2idx, words
import random

# from src.util.es_utils import _analyze
from config import data_dir
from src.util.utils import get_pkl

random.seed(0)


def pop_to_words(pop: list):
    pop_words = [words[i] for i, p in enumerate(pop) if p == 1]
    return pop_words


def idx_2pop(d_idx: list, s_words: list):
    pop_i = [0] * len(word2idx)
    for di in d_idx:
        di_w = s_words[di]
        if di_w in word2idx:
            w_idx = word2idx[di_w]
            pop_i[w_idx] = 1
    return pop_i


def Create(s_words: list, size=10):
    '''
    创建种群，大小取决于sentence的分词长度
    :param sentence:
    :return:
    '''
    # s_words = _analyze([sentence])[0]
    pops = []
    temp = []
    if len(s_words) <= 3:
        return None
    r = list(range(2, len(s_words)))
    r_idx = list(range(len(s_words)))
    count = 0

    # 不删的
    d_idx = list(range(len(s_words)))
    pop_i = idx_2pop(d_idx, s_words)
    pop_words = pop_to_words(pop_i)
    pop_sent = ','.join(pop_words)
    temp.append(pop_sent)
    pops.append(pop_i)
    # 删一个的
    for i in range(len(s_words)):
        d_idx = list(range(len(s_words)))
        d_idx.pop(i)
        pop_i = idx_2pop(d_idx, s_words)
        pop_words = pop_to_words(pop_i)
        pop_sent = ','.join(pop_words)
        if pop_sent not in temp:
            temp.append(pop_sent)
            pops.append(pop_i)

    while len(temp) < size * 2:
        count += 1
        if count > size ** 3:
            break
        # 选几个
        num = random.sample(r, 1)
        d_idx = random.sample(r_idx, num[0])

        pop_i = [0] * len(word2idx)
        for di in d_idx:
            di_w = s_words[di]
            if di_w in word2idx:
                w_idx = word2idx[di_w]
                pop_i[w_idx] = 1
        if len(pop_i) >= 3:
            pop_words = pop_to_words(pop_i)
            pop_sent = ','.join(pop_words)
            if pop_sent not in temp:
                temp.append(pop_sent)
                pops.append(pop_i)
    fp = pops[0].copy()
    pops.pop(0)
    random.shuffle(pops)
    return [fp] + pops[:size - 1]


if __name__ == '__main__':
    words = get_pkl(path=os.path.join(data_dir, 'train_words.pkl'))
    print(len(words))
    pop = Create(words[0])
    for p in pop:
        print(p)
    # pass
