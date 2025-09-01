#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 11:50
# @Author  : hukangzhe
# @File    : embedder.py
# @Description :

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.embedding_model.encode(texts, batch_size=batch_size, convert_to_numpy=True)

