#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import requests
import numpy as np
import os
import json
from typing import List
from collections import Counter
from config import data_csv, data_dir
from src.strategy.Create import Create
from src.util.utils import save_pkl, get_pkl
from src.util.ga_utils import word2idx
from tqdm import tqdm
from src.logers import LOGS

index = 'bs_index'
host = 'http://0.0.0.0:8080/es/v0'


# test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'), sep='\t')
# test_texts = test_data['text'].values.tolist()
# test_ids = test_data['subject_id'].values.tolist()


def resetbm25(k, b):
    data = {
        "b": b,
        "k": k,
        "index": index
    }
    res = requests.post(f'{host}/set_bm25', data=json.dumps(data))
    return res


def most_id(ids, items_score):
    if ids[0] == ids[3]:
        return ids[0]
    c = Counter(ids)
    mc = c.most_common(1)
    if len(items_score) > 1 and items_score[0] - 5 > items_score[1]:
        return ids[0]
    else:
        return mc[0][0]


def predict(sentences: List[str]):
    server_max_batch_size = 2000
    if len(sentences) > server_max_batch_size:
        batch_sentences = []
        for i in range(0, len(sentences), server_max_batch_size):
            batch_sentences.append(sentences[i:i + server_max_batch_size])
    else:
        batch_sentences = [sentences]

    result_ids = []
    for sentences in batch_sentences:
        url = 'http://0.0.0.0:8080/es/v0/search'
        source = {
            "texts": sentences,
            'size': 10,
            'index': index
        }
        res = requests.post(url=url, data=json.dumps(source), )
        res_json = res.json()
        data = res_json['data']
        result_id = []
        for si_data in data:
            si_hits = si_data['hits']
            sorted_hits = sorted(si_hits, key=lambda k: k['_score'], reverse=True)
            items_subject = [item['_source']['subject_id'] for item in sorted_hits]
            # sim_sents = [item['_source']['text'] for item in sorted_hits]
            items_score = [item['_score'] for item in sorted_hits]
            predict_id = most_id(items_subject, items_score)
            result_id.append(predict_id)
        result_ids.extend(result_id)
    return result_ids


def search(all_sentences: List, all_subject_ids: List):
    '''
    搜索和sentence最相似的tok 30，取其中subject_id和输入一样的（同类，命中），的分数和
    :param sentence:
    :param subject_ids:
    :return:
    '''
    if len(all_sentences) == 0:
        return []
    server_max_batch_size = 5000
    if len(all_sentences) > server_max_batch_size:
        batch_sentences = []
        batch_subject_ids = []
        for i in range(0, len(all_sentences), server_max_batch_size):
            batch_sentences.append(all_sentences[i:i + server_max_batch_size])
            batch_subject_ids.append(all_subject_ids[i:i + server_max_batch_size])
    else:
        batch_sentences = [all_sentences]
        batch_subject_ids = [all_subject_ids]

    result_s = []
    for bi in range(len(batch_sentences)):
        bi_sentences = batch_sentences[bi]
        url = 'http://0.0.0.0:8080/es/v0/search'
        source = {
            "texts": bi_sentences,
            'size': 30,
            'index': index
        }
        try:
            res = requests.post(url=url, data=json.dumps(source), )
            res_json = res.json()
            data = res_json['data']
            scores = []
            for j, si_data in enumerate(data):
                subject_id = batch_subject_ids[bi][j]
                try:
                    si_hits = si_data['hits']
                    items_scores = [item['_score'] for item in si_hits if item['_source']['subject_id'] == subject_id]
                    score = sum(items_scores)
                except Exception as e:
                    score = 0
                    print(e)
                scores.append(score)
        except Exception as e:
            LOGS.log.debug(f'{e, bi_sentences}')
            scores = [0] * len(bi_sentences)

        result_s.extend(scores)
    return result_s


def test(k, b):
    global test_texts, test_ids
    resetbm25(k, b)
    predict_ids = predict(test_texts)
    res = [1 if str(test_ids[i]) == str(predict_ids[i]) else 0 for i in range(len(test_ids))]
    accuracy = sum(res) * 100.0 / len(test_ids)
    return accuracy


def build_es_train_data():
    train_pd = pd.read_csv(os.path.join(data_dir, 'train.csv'), sep='\t')
    texts = train_pd['text'].values.tolist()
    subject_ids = train_pd['subject_id'].values.tolist()

    all_pops = []
    all_sentences = []
    all_subject_ids = []
    all_sen_idx = []
    print('加载。。。')
    all_words = get_pkl(path=os.path.join(data_dir, 'train_words.pkl'))
    print(len(all_words))
    for i in tqdm(range(len(texts))):
        s_words = all_words[i]
        s_s_words = [w for w in s_words if w in word2idx]
        if len(s_s_words) <= 2: continue
        res = Create(s_words=s_s_words)
        _sentences = [','.join(r[1]) for r in res]
        _ids = [str(subject_ids[i])] * len(_sentences)
        sen_idx = [i] * len(_sentences)
        all_sen_idx.extend(sen_idx)
        # all_pops.extend(pops)
        all_sentences.extend(_sentences)
        all_subject_ids.extend(_ids)
    print('结束加载。。。')
    print('预测：', len(all_sentences))
    save_pkl(obj=(all_sentences, all_subject_ids), path=os.path.join(data_dir, 'all_search_data.pkl'))
    all_scores = search(all_sentences, all_subject_ids)
    all_sentence_data = [all_sen_idx, all_sentences, all_subject_ids, all_scores]
    save_pkl(obj=all_sentence_data, path=os.path.join(data_dir, 'all_sentence_data.pkl'))


if __name__ == '__main__':
    ws = ['虎牌', '颈肩', '舒', '50g', '虎标', '颈肩', '舒']
    ws2 = ws[::-1]
    s = ','.join(ws)
    s2 = ','.join(ws2)
    # 表面词序不影响分数
    print(search([s, s2], all_subject_ids=['143491', '143491']))
    # print(test(k=1.2, b=0.75))
    # build_es_train_data()
