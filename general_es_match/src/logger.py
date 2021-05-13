import inspect
import json
import pathlib
import re
import sys
import typing

import colorama
from loguru import logger as original_logger

colorama.init()


class StructuredLogger:
    def __init__(self):

        self.o_logger = original_logger
        self.o_logger.level("ACTION", no=20, color="<blue>")
        self.init = False

    def check_init(func):
        """装饰器，用来检查logger是否初始化

        Returns:

        """

        def wrapper(self, msg, *args, **kwargs):
            if not self.init:
                logger.error("logger必须在初始化之后使用")
                raise RuntimeError("必须init_log之后调用")
            # 获取调用信息
            callerframerecord = inspect.stack()[1]
            frame = callerframerecord[0]
            info = inspect.getframeinfo(frame)
            file_name = re.sub(r"\.py", "", pathlib.Path(info.filename).name)
            info_str = f"{file_name}.{info.function}:{info.lineno}"
            info_str = info_str.replace(".<module>", "")
            # msg = f"{info_str:28} | {msg}"
            return func(self, msg, *args, info_str=info_str, **kwargs)

        return wrapper

    @staticmethod
    def prepare_log(bind_variables, msg, info_str):
        if bind_variables is None:
            bind_variables = {}
        main_info = ""
        if "uid" in bind_variables:
            if main_info:
                main_info = f"{main_info},U:{bind_variables['uid']}"
            else:
                main_info = f"U:{bind_variables['uid']}"
        if "turn" in bind_variables:
            if main_info:
                main_info = f"{main_info},T:{bind_variables['turn']}"
            else:
                main_info = f"T:{bind_variables['turn']}"
        if "dialogue_state" in bind_variables:
            if main_info:
                main_info = f"{main_info},S:{bind_variables['dialogue_state']}"
            else:
                main_info = f"S:{bind_variables['dialogue_state']}"
        if main_info:
            msg = f"{main_info} {msg}"
        if info_str:
            msg = f"{info_str:28} | {msg:10}"
        return bind_variables, msg

    @check_init
    def action(self, msg, bind_variables=None, info_str=None):
        """一般性操作类信息

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).log("ACTION", msg)

    @check_init
    def info(self, msg, bind_variables=None, info_str=None):
        """一般性信息

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).info(msg)

    @check_init
    def warn(self, msg, bind_variables=None, info_str=None):
        """仅用于提示危险操作或可能触发错误的操作

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).warning(msg)

    @check_init
    def debug(self, msg, bind_variables=None, info_str=None):
        """仅用于打印调试信息时使用

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).debug(msg)

    @check_init
    def critical(self, msg, bind_variables=None, info_str=None):
        """提示重要信息时使用

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).critical(msg)

    @check_init
    def error(self, msg, bind_variables=None, info_str=None):
        """程序抛出错误/异常或者即将遇到错误/异常时使用

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).error(msg)

    @check_init
    def success(self, msg, bind_variables=None, info_str=None):
        """用于对话引擎或者相关组件成功加载/调用/建立时使用

        Args:
            msg:
            bind_variables:

        Returns:

        """
        bind_variables, msg = self.prepare_log(bind_variables, msg, info_str)
        self.o_logger.bind(**bind_variables).success(msg)


def format_func(d):
    if d["extra"]:
        extra_str = json.dumps(d["extra"], ensure_ascii=False)
    else:
        extra_str = ""
    head_str = f" | {d['level']:7} | {d['message']}"
    result = head_str + " | " + extra_str + "\n"
    result = "{time:YYYY-MM-DD-HH:mm:ss.SSS}" + result.replace("{", "{{").replace(
        "}", "}}"
    )
    return result


logger = StructuredLogger()


def init_log(
    log_dir=None,
    file_name=None,
    console_output=False,
    only_console=False,
    console_show_all=True,
):
    global logger
    config = {
        "handlers": [],
    }
    if not only_console:
        if log_dir is None or file_name is None:
            raise RuntimeError("没有指定日志路径或者文件名")
        file_name = re.sub(r"\.log", "", file_name)
        log_dir = pathlib.Path(log_dir)
        if not log_dir.exists():
            log_dir.mkdir()
        config["handlers"].append(
            {
                "sink": f"{str(log_dir / f'{file_name}.log')}",
                "format": format_func,
                "rotation": "1:00",
                # fixme：gunicorn 下不可以将enqueue设置为True，否则会出现超时错误，原因未知
                "enqueue": False,
            },
        )
    if console_output:
        config["handlers"].append(
            {
                "sink": sys.stdout,
                "format": format_func
                if console_show_all
                else "<green>{time:YYYY-MM-DD-HH:mm:ss.SSS}</green> | <level>{level:7}</level> | <level>{message}</level>",
                "colorize": True,
                # fixme：gunicorn 下不可以将enqueue设置为True，否则会出现超时错误，原因未知
                "enqueue": False,
            }
        )
    logger.o_logger.configure(**config)
    logger.init = True
    logger.success("初始化结构化logger成功")
