#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/13 下午4:18
# @File    : logers.py


from loguru import logger
import os


class LOGS:
    log = logger

    @classmethod
    def init(cls, log_file):
        sample_file = log_file.replace('.log', '_sample.log')
        if os.path.exists(log_file):
            os.remove(log_file)
        if os.path.exists(sample_file):
            os.remove(sample_file)

        cls.log.add(log_file)
        cls.log.add(sample_file, filter=lambda x: '[sample]' in x['message'])
