#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/25 19:51
# @Author  : hukangzhe
# @File    : logger.py.py
# @Description : 日志配置模块
import logging
import sys


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.FileHandler("storage/logs/app.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

