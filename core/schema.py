#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 21:11
# @Author  : hukangzhe
# @File    : schema.py
# @Description : 不直接处理纯文本字符串. 引入一个标准化的数据结构来承载文本块，既有内容，也有元数据。

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk(Document):
    parent_id: int = None