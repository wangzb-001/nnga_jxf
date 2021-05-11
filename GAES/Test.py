# !/usr/bin/python3
# 名称：遗传算法
# 时间: 2018/12/14 15:28
# 作者:jiangxinfa
# 邮件:425776024@qq.com
import src.problems.ES as ES
from src.problems.GeneticAlgorithm import *
from tqdm import tqdm
from src.util.utils import get_pkl
import torch
import os
from config import data_dir
import pandas as pd
from src.nn.cross_ga_nn import CrossGaModel
from src.logers import LOGS

seed = 0

acc_rate = 0.001
np.random.seed(seed)
torch.manual_seed(seed)

es = ES.ES()

if __name__ == '__main__':
    times = 1
    LOGS.init('log_test3/eslog.log')
    cross_rate, mutation_rate = 0.5, 0.1
    i = 0
    CrossGaModel.init(dim=es.valuable_num, test_func=es)
    CrossGaModel.load_model('model/model_3_90_.pt')
    CrossGaModel.model.eval()

    train_pd = pd.read_csv(os.path.join(data_dir, 'train.csv'), sep='\t')
    texts = train_pd['text'].values.tolist()
    # 100条数据
    start = 101
    end = 201
    subject_ids = train_pd['subject_id'].values.tolist()

    all_words = get_pkl(path=os.path.join(data_dir, 'train_words.pkl'))

    subject_ids = subject_ids[start:end]
    all_words = all_words[start:end]
    ga = GeneticAlgorithm(cross_rate, mutation_rate, es)
    for i in tqdm(range(len(all_words))):
        ws = all_words[i]
        es.word = ws
        res = ga.test(words=ws, subject_id=subject_ids[i])
        LOGS.log.debug(f'[sample]res:{res}')
        LOGS.log.debug('[sample]\n')
