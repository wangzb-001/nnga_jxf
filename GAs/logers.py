#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 下午4:09
# @File    : logers.py

from loguru import logger
import os


class LOGS:
    log = logger
    pre = None

    @classmethod
    def init(cls, log_file):
        if os.path.exists(log_file):
            os.remove(log_file)
        if cls.pre:
            cls.log.remove(cls.pre)
        cls.pre = cls.log.add(log_file)
        cls.log.debug('this is a debug message')
