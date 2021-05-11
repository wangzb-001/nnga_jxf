#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/7 5:44 下午
# @File    : es_upload.py
import os
import requests
import pandas as pd
import json
from tqdm import tqdm
from src.util.utils import get_pkl
from config import data_csv, data_dir
from sklearn.model_selection import train_test_split
from collections import Counter
import random
import numpy as np

random.seed(0)
np.random.seed(0)

index = 'bs_index'

X_train, X_test, y_train, y_test = [], [], [], []


def load_data():
    train_df = pd.read_csv(data_csv, sep='\t', names=['text', 'subject_id'])

    texts = train_df['text'].values
    labels = train_df['subject_id'].values

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def upload_train():
    url = 'http://0.0.0.0:8080/es/v0/insert_texts'
    bodys = []
    for i in range(len(X_train)):
        body = {
            'text': X_train[i],
            'subject_id': str(y_train[i]),
            'index': i
        }
        bodys.append(body)

    es_body = {
        "index": index,
        "bodys": bodys
    }
    # 训练数据上传
    res = requests.post(url=url, data=json.dumps(es_body))
    print(res.json())


def save(X, Y, name='test.csv'):
    test_df = pd.DataFrame()
    test_df['text'] = X
    test_df['subject_id'] = Y
    test_df.to_csv(os.path.join(data_dir, name), sep='\t', index=False)


def _analyze(texts:list):
    url = 'http://0.0.0.0:8080/es/v0/_analyze'
    body = {
        "index": "bs_index",
        "texts": texts,
        "analyzer": "ik_smart"
    }
    res = requests.post(url=url, data=json.dumps(body))
    data = []
    if res.status_code == 200:
        res = res.json()
        for item in res:
            tokens = item['tokens']
            words = [token['token'] for token in tokens]
            data.append(words)
    return data


def analyze(step=1000):
    words = []
    for i in tqdm(range(0, len(X_train), step)):
        if i + step < len(X_train):
            di = X_train[i: i + step].tolist()
        else:
            di = X_train[i: len(X_train)].tolist()
        di_words = _analyze(texts=di)
        words.extend(di_words)
    return words


def write_word_count(words, path: str):
    f_words = [x for tup in words for x in tup]
    words_counter = Counter(f_words)
    words_counter = words_counter.most_common()
    # 词频最少10个
    min_freq = 10
    m_words = [w for w in words_counter if w[1] > min_freq]

    with open(path, mode='w', encoding='utf-8') as wf:
        for item in m_words:
            wf.write(f"{item[0]}\t{item[1]}\n")


if __name__ == '__main__':
    # save(X_test, y_test, 'test.csv')
    # save(X_train, y_train, 'train.csv')
    # words = analyze()
    # save_pkl(path=os.path.join(data_dir, 'train_words.pkl'), obj=words)
    # words = get_pkl(path=os.path.join(data_dir, 'train_words.pkl'))
    # print(len(words))
    # write_word_count(words, os.path.join(data_dir, 'vocab.txt'))
    # print(_analyze(['信龙,硼酸,250ml*1,瓶/盒,用于,冲洗,伤口,消毒,硼酸,洗液']))
    # print(_analyze(['用于,冲洗,伤口,消毒,硼酸,洗液']))
    # print(_analyze(['用于冲洗伤口消毒,硼酸洗液']))
    pass