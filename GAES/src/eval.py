#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/8 6:17 下午
# @File    : eval.py

# 评估 test.csv 中的准确性
import pandas as pd
from config import test_csv
import random
import numpy as np

random.seed(0)
np.random.seed(0)
test_df = pd.read_csv(test_csv, sep='\t', names=['text', 'subject_id'])
