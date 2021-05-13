#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import APIRouter
from .base import *

show_router = APIRouter()

'''
展示类路由
'''


@show_router.get("/get_indexes")
def get_indexes():
    """
    获得所有注册的索引
    Returns:
    """
    try:
        indexs = ES_Model.get_indexs()
        return success_response({"indexs": indexs})
    except Exception as e:
        return error_response(e)
