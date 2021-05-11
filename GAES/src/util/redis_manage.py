#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/1/2 3:31 下午
# @File    : redis_manage.py
import pickle
import typing
import redis

redis_host = '0.0.0.0'
redis_port = 6379
redis_password = None
redis_db = 5


class RedisManage:
    r: redis.Redis = None

    @classmethod
    def init(cls):
        redis_pool = redis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
        )
        redis_connection = redis.Redis(connection_pool=redis_pool, charset='utf-8')
        cls.set_connection(r=redis_connection)

    @classmethod
    def set_connection(
            cls,
            r: redis.Redis = None,
            host: str = "localhost",
            port: int = 6379,
            db: int = 0,
            password: str = None,
    ) -> None:
        """
            配置一个redis连接
        Args:
            password:
            db:
            port:
            host:
            r:

        Returns:

        """

        if r:
            cls.r = r
        else:
            cls.r = redis.Redis(host=host, port=port, db=db, password=password, charset='utf-8')

    @classmethod
    def keys(cls, r: redis.Redis = None) -> typing.List[bytes]:
        if r is None:
            r = cls.r
        return r.keys()

    @classmethod
    def exist(cls, name: str, r: redis.Redis = None, time_e=10) -> bool:
        if r is None:
            r = cls.r
        return bool(r.exists(name))

    @classmethod
    def set(cls, name, value=None, binary_object=None, r: redis.Redis = None, time_e=60 * 60 * 8) -> bool:
        if cls.r is None:
            cls.init()
        if r is None:
            r = cls.r
        if value is None and binary_object is None:
            raise RuntimeError("value 和 binary_object 至少赋值一个")
        if binary_object:
            res = r.set(name, pickle.dumps(value))
        else:
            res = r.set(name, value)
        r.expire(name, time_e)
        return res

    @classmethod
    def get(cls, name, is_binary_object=False, r: redis.Redis = None) -> typing.Any:
        if cls.r is None:
            cls.init()
        if r is None:
            r = cls.r
        value = r.get(name)
        if value and is_binary_object:
            s = pickle.loads(value)
            return s
        else:
            return value

    @classmethod
    def delete(cls, name, r: redis.Redis = None) -> typing.Any:
        if r is None:
            r = cls.r
        return r.delete(name)


def get_user_verycode(user_name):
    # 获得用户验证码
    redis_code = RedisManage.get(user_name)
    if redis_code is None:
        return None
    if isinstance(redis_code, bytes):
        redis_code = redis_code.decode(encoding='utf-8')
    redis_code = redis_code.lower()
    return redis_code


if __name__ == '__main__':
    RedisManage.init()
    RedisManage.set('j', '斤斤计较')
    jxf = RedisManage.get('j')
    if isinstance(jxf, bytes):
        jxf = jxf.decode(encoding='utf-8')
    print(jxf)
