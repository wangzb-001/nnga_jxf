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
    LOGS.init('log_train1/eslog.log')
    cross_rate, mutation_rate = 0.5, 0.1
    i = 0
    CrossGaModel.init(dim=es.valuable_num, test_func=es)
    # CrossGaModel.load_model('model/model_50_.pt')
    train_pd = pd.read_csv(os.path.join(data_dir, 'train.csv'), sep='\t')
    texts = train_pd['text'].values.tolist()

    start = 0
    end = 100
    subject_ids = train_pd['subject_id'].values.tolist()

    all_words = get_pkl(path=os.path.join(data_dir, 'train_words.pkl'))

    subject_ids = subject_ids[start:end]
    all_words = all_words[start:end]
    ga = GeneticAlgorithm(cross_rate, mutation_rate, es)
    epoch = 5
    for e in range(epoch):
        for i in tqdm(range(len(all_words))):
            if i > 1 and i % 30 == 0:
                torch.save(CrossGaModel.model.state_dict(), f'model/model_{e}_{i}_.pt')
            ws = all_words[i]
            es.word = ws
            res = ga.run(words=ws, subject_id=subject_ids[i])
