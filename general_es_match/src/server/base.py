#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xinfa.jiang
# @File    : base.py
from boltons import tbutils
from logger import logger
from es.ES_Model import ES_Model


def success_response(data):
    return {"status": 200, "msg": "服务正常", "data": data}


def error_response(e):
    exc_info = tbutils.ExceptionInfo.from_current()
    logger.error(
        "服务异常",
        {"traceback": exc_info.to_dict(), "status": 500, "error_doc": str(e.__doc__), },
    )
    return {"status": 500, "msg": f"error:{e.__class__},args:{e.args},{e.__doc__}"}
