#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/27 19:52
# @Author  : hukangzhe
# @File    : retriever.py
# @Description : 负责向量化、存储、检索的模块
import os
import faiss
import numpy as np
import pickle
import logging
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
from .schema import Document, Chunk


class HybridVectorStore:
    def __init__(self, config: dict, embedder):
        self.config = config["vector_store"]
        self.embedder = embedder
        self.faiss_index = None
        self.bm25_index = None
        self.parent_docs: Dict[int, Document] = {}
        self.child_chunks: List[Chunk] = []

    def build(self, parent_docs: Dict[int, Document], child_chunks: List[Chunk]):
        self.parent_docs = parent_docs
        self.child_chunks = child_chunks

        # Build Faiss index
        child_text = [child.text for child in child_chunks]
        embeddings = self.embedder.embed(child_text)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(embeddings)
        logging.info(f"FAISS index built with {len(child_chunks)} vectors.")

        # Build BM25 index
        tokenize_chunks = [doc.text.split(" ") for doc in child_chunks]
        self.bm25_index = BM25Okapi(tokenize_chunks)
        logging.info(f"BM25 index built for {len(child_chunks)} documents.")

        self.save()

    def search(self, query: str, top_k: int , alpha: float) -> List[Tuple[int, float]]:
        # Vector Search
        query_embedding = self.embedder.embed([query])
        distances, indices = self.faiss_index.search(query_embedding, k=top_k)
        vector_scores = {idx : 1.0/(1.0 + dist) for idx, dist in zip(indices[0], distances[0])}

        # BM25 Search
        tokenize_query = query.split(" ")
        bm25_scores = self.bm25_index.get_scores(tokenize_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_scores = {idx: bm25_scores[idx] for idx in bm25_top_indices}

        # Hybrid Search
        all_indices = set(vector_scores.keys()) | set(bm25_scores.keys()) # 求并集
        hybrid_scors = {}

        # Normalization
        max_v_score = max(vector_scores.values()) if vector_scores else 1.0
        max_b_score = max(bm25_scores.values()) if bm25_scores else 1.0
        for idx in all_indices:
            v_score = (vector_scores.get(idx, 0))/max_v_score
            b_score = (bm25_scores.get(idx, 0))/max_b_score
            hybrid_scors[idx] = alpha * v_score + (1 - alpha) * b_score

        sorted_indices = sorted(hybrid_scors.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return sorted_indices

    def get_chunks(self, indices: List[int]) -> List[Chunk]:
        return [self.child_chunks[i] for i in indices]

    def get_parent_docs(self, chunks: List[Chunk]) -> List[Document]:
        parent_ids = sorted(list(set(chunk.parent_id for chunk in chunks)))
        return [self.parent_docs[pid] for pid in parent_ids]

    def save(self):
        index_path = self.config['index_path']
        metadata_path = self.config['metadata_path']

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        logging.info(f"Saving FAISS index to: {index_path}")
        try:
            faiss.write_index(self.faiss_index, index_path)
        except Exception as e:
            logging.error(f"Failed to save FAISS index: {e}")
            raise

        logging.info(f"Saving metadata data to: {metadata_path}")
        try:
            with open(metadata_path, 'wb') as f:
                metadata = {
                    'parent_docs': self.parent_docs,
                    'child_chunks': self.child_chunks,
                    'bm25_index': self.bm25_index
                }
                pickle.dump(metadata, f)
        except Exception as e:
            logging.error(f"Failed to save metadata: {e}")
            raise

        logging.info("Vector store saved successfully.")

    def load(self) -> bool:
        """
        从磁盘加载整个向量存储状态,成功时返回 True，失败时返回 False。
        """
        index_path = self.config['index_path']
        metadata_path = self.config['metadata_path']

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logging.warning("Index files not found. Cannot load vector store.")
            return False

        logging.info(f"Loading vector store from disk...")
        try:
            # Load FAISS index
            logging.info(f"Loading FAISS index from: {index_path}")
            self.faiss_index = faiss.read_index(index_path)

            # Load metadata
            logging.info(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.parent_docs = metadata['parent_docs']
                self.child_chunks = metadata['child_chunks']
                self.bm25_index = metadata['bm25_index']

            logging.info("Vector store loaded successfully.")
            return True

        except Exception as e:
            logging.error(f"Failed to load vector store from disk: {e}")
            self.faiss_index = None
            self.bm25_index = None
            self.parent_docs = {}
            self.child_chunks = []
            return False

















