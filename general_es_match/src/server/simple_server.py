#!/usr/bin/env python
# -*- coding: utf-8 -*-

from fastapi import APIRouter

simple_router = APIRouter()


@simple_router.get('/')
async def index():
    return {"message": 'success'}
