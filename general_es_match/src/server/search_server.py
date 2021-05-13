#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import APIRouter
from .base import success_response

search_router = APIRouter()


@search_router.post("/search")
def search():
    return success_response('search')
