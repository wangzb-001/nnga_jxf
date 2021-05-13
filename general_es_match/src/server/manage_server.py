#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import APIRouter
from .base import success_response, error_response

manage_router = APIRouter()
'''
管理类路由:增删
'''


@manage_router.post("/index_delete")
def index_delete():
    """
    接受参数，下线指定的索引
    """
    try:
        return success_response("index_offline")
    except Exception as e:
        return error_response(e)


@manage_router.post("/index_add")
def index_add():
    """接受输入参数，创建实例，更新本地实例文件，根据参数指示决定是否立刻上线
    Returns:
    """
    return {"index_id": 1, "index_state": "online"}


@manage_router.post("/data_add")
def data_insert():
    try:
        return success_response("data_add")
    except Exception as e:
        return error_response(e)
