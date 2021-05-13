#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xinfa.jiang
# @File    : __init__.py.py

from server.simple_server import simple_router
from server.show_server import show_router
from server.search_server import search_router
from server.manage_server import manage_router

__author__ = 'Jiang.XinFa'

__routers__ = ["simple_router", "add_router", "show_router", "search_router", "manage_router"]

__all__ = ["simple_router", "show_router", "search_router", "manage_router"]
