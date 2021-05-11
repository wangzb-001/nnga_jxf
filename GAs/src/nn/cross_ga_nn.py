#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 下午5:03
# @File    : cross_ga_nn.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from problems.Problem_Abs import Problem_Abs


class SelfAttention(nn.Module):

    def __init__(self, d_hid, dropout=0.1):
        super().__init__()
        self.scorer = nn.Linear(d_hid, d_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        # 1 2 101
        batch_size, seq_len, feature_dim = input_seq.size()
        # 1 x 2
        scores = self.scorer(input_seq.contiguous().view(-1, feature_dim))
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(0).expand_as(input_seq).mul(input_seq).sum(1)
        context = self.dropout(context)
        # 1 x 101
        return context


class Net(nn.Module):
    def __init__(self, dim, test_func: Problem_Abs):
        '''
        :param dim:问题解 的维度
        '''
        self.dim = dim
        self.test_func = test_func
        super(Net, self).__init__()
        self.fc_input1 = nn.Linear(dim + 1, dim + 1)
        self.fc_input2 = nn.Linear(dim + 1, dim + 1)

        # self.fc_input_fv = nn.Linear(2, 2)
        self.activation = 'softmax'
        self.attention = SelfAttention(dim + 1)
        self.fc_out = nn.Linear(2 * (dim + 1), dim)
        # _init_weight初始化权重，也可以不用，照样可以效果好，用了更好
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():  # 继承nn.Module的方法
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight)
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, X):
        input1 = X[0]
        input2 = X[1]
        # input_fv = X[2]
        if type(input1) == np.ndarray:
            input1 = torch.tensor(input1, dtype=torch.float)
            input2 = torch.tensor(input2, dtype=torch.float)
        # x_3 = torch.mean(x_1.add(x_2), dim=0).reshape_as(x_1)
        if self.activation == 'softmax':
            x_1 = F.softmax(self.fc_input1(input1), dim=1)
            x_2 = F.softmax(self.fc_input2(input2), dim=1)
            # input_fv = F.softmax(self.fc_input_fv(input_fv), dim=1)
        elif self.activation == 'sigmoid':
            x_1 = F.sigmoid(self.fc_input1(input1))
            x_2 = F.sigmoid(self.fc_input2(input2))
        elif self.activation == 'tanh':
            x_1 = F.tanh(self.fc_input1(input1))
            x_2 = F.tanh(self.fc_input2(input2))
        elif self.activation == 'relu':
            x_1 = F.relu(self.fc_input1(input1))
            x_2 = F.relu(self.fc_input2(input2))
        elif self.activation == 'gelu':
            x_1 = F.gelu(self.fc_input1(input1))
            x_2 = F.gelu(self.fc_input2(input2))
        else:
            x_1 = self.fc_input1(input1)
            x_2 = self.fc_input2(input2)
        # 2 x 1
        # input_fv = input_fv.view(2, -1)
        # 2 x 100
        # x_3 = torch.vstack((x_1, x_2))
        # 1 x 2x 101
        x_1 = x_1.unsqueeze(0)
        x_2 = x_2.unsqueeze(0)
        #
        context1 = self.attention(x_1)
        context2 = self.attention(x_2)
        x_3 = torch.hstack((context1, context2))
        x3 = self.fc_out(x_3)
        # x3 = torch.clamp(x3, min=self.test_func.Bound[0], max=self.test_func.Bound[1])
        if torch.any(torch.isinf(x3)):
            x3 = torch.where(torch.isinf(x3), torch.full_like(x3, self.test_func.Bound[1]), x3)
        if torch.any(torch.isnan(x3)):
            x3 = torch.where(torch.isnan(x3), torch.full_like(x3, 0), x3)

        return x3


class CrossGaModel:
    model: nn.Module = None
    optimizer = None
    criterion = None
    dim = -1
    normalize = False

    @classmethod
    def init(cls, test_func, normalize=False, dim=5):
        cls.dim = dim
        cls.normalize = normalize
        cls.model = Net(dim=dim, test_func=test_func)
        cls.optimizer, cls.criterion = optim.SGD(cls.model.parameters(), lr=0.1), nn.MSELoss(reduction='sum')

    @classmethod
    def call_loss(cls, input1, input2):
        if type(input1) == np.ndarray:
            input1 = torch.tensor(input1, dtype=torch.float)
        if type(input2) == np.ndarray:
            input2 = torch.tensor(input2, dtype=torch.float)
        cls.model.train(mode=False)
        loss = cls.criterion(input1, input2)
        return loss

    @classmethod
    def train_one(cls, input1, input2, target, func, times=1):
        '''
        训练一次
        :param input1: 两个个体解之1
        :param input2: 两个个体解之1
        :param target: 两个个体解的新的更优解
        :return:
        '''
        i1f = func(input1)
        i2f = func(input2)
        if isinstance(i1f, list) or isinstance(i1f, np.ndarray):
            if len(i1f.shape) == 1 and i1f.shape[0] == 1:
                i1f = i1f[0]
            else:
                i1f = i1f.tolist()
        if isinstance(i2f, list) or isinstance(i2f, np.ndarray):
            if len(i2f.shape) == 1 and i2f.shape[0] == 1:
                i2f = i2f[0]
            else:
                i2f = i2f.tolist()
        if type(input1) == np.ndarray:
            if len(input1.shape) == 1:
                input1 = [input1]
            input1 = torch.tensor(input1, dtype=torch.float)
        if type(input2) == np.ndarray:
            if len(input2.shape) == 1:
                input2 = [input2]
            input2 = torch.tensor(input2, dtype=torch.float)
        if type(target) == np.ndarray:
            if len(target.shape) == 1:
                target = [target]
            target = torch.tensor(target, dtype=torch.float)
        fv1 = torch.tensor([[i1f]], dtype=torch.float)
        fv2 = torch.tensor([[i2f]], dtype=torch.float)
        input1 = torch.hstack((input1, fv1))
        input2 = torch.hstack((input2, fv2))
        if cls.normalize:
            input1 = F.normalize(input1, p=2, dim=1)
            input2 = F.normalize(input2, p=2, dim=1)
            # input_fv = F.normalize(input_fv, p=2, dim=1)
            # v_input_fv = F.normalize(v_input_fv, p=2, dim=1)
            # target不能归一化，否则怎么整理输出结果？
            # target = F.normalize(target, p=2, dim=1)

        # 训练模式
        cls.model.train(mode=True)
        loss = 0
        for _ in range(times):
            # data = [[input1, input2, input_fv], [input2, input1, v_input_fv]]
            data = [[input1, input2]]
            for di in data:
                # 在训练的迭代中：
                cls.optimizer.zero_grad()  # 清零梯度缓存
                output = cls.model(di)
                _loss = cls.criterion(output, target)
                _loss.backward()
                cls.optimizer.step()  # 更新参数
                loss += _loss

        return loss

    @classmethod
    def predict_one(cls, input1, input2, func):
        i1f = func(input1)
        i2f = func(input2)
        if isinstance(i1f, list) or isinstance(i1f, np.ndarray):
            if len(i1f.shape) == 1 and i1f.shape[0] == 1:
                i1f = i1f[0]
            else:
                i1f = i1f.tolist()
        if isinstance(i2f, list) or isinstance(i2f, np.ndarray):
            if len(i2f.shape) == 1 and i2f.shape[0] == 1:
                i2f = i2f[0]
            else:
                i2f = i2f.tolist()
        if type(input1) == np.ndarray:
            if len(input1.shape) == 1:
                input1 = [input1]
            input1 = torch.tensor(input1, dtype=torch.float)
        if type(input2) == np.ndarray:
            if len(input2.shape) == 1:
                input2 = [input2]
            input2 = torch.tensor(input2, dtype=torch.float)
        fv1 = torch.tensor([[i1f]], dtype=torch.float)
        fv2 = torch.tensor([[i2f]], dtype=torch.float)
        input1 = torch.hstack((input1, fv1))
        input2 = torch.hstack((input2, fv2))
        if cls.normalize:
            input1 = F.normalize(input1, p=2, dim=1)
            input2 = F.normalize(input2, p=2, dim=1)
        cls.model.train(mode=False)
        ipt = [input1, input2]
        output = cls.model(ipt)
        return output
