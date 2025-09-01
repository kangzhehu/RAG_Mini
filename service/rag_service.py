#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/30 11:50
# @Author  : hukangzhe
# @File    : rag_service.py
# @Description :
import logging
import os
from typing import List, Generator, Tuple
from core.schema import Document
from core.embedder import EmbeddingModel
from core.loader import MultiDocumentLoader
from core.splitter import HierarchicalSemanticSplitter
from core.vector_store import HybridVectorStore
from core.llm_interface import LLMInterface


class RAGService:
    def __init__(self, config: dict):
        self.config = config
        logging.info("Initializing RAG Service...")
        self.embedder = EmbeddingModel(config['models']['embedding'])
        self.vector_store = HybridVectorStore(config, self.embedder)
        self.llm = LLMInterface(config)
        self.is_ready = False   # 是否准备好进行查询
        logging.info("RAG Service initialized. Knowledge base is not loaded.")

    def load_knowledge_base(self) -> Tuple[bool, str]:
        """
        尝试从磁盘加载
        Returns:
            A tuple (success: bool, message: str)
        """
        if self.is_ready:
            return True, "Knowledge base is already loaded."

        logging.info("Attempting to load knowledge base from disk...")
        success = self.vector_store.load()
        if success:
            self.is_ready = True
            message = "Knowledge base loaded successfully from disk."
            logging.info(message)
            return True, message
        else:
            self.is_ready = False
            message = "No existing knowledge base found or failed to load. Please build a new one."
            logging.warning(message)
            return False, message

    def build_knowledge_base(self, file_paths: List[str]) -> Generator[str, None, None]:
        self.is_ready = False
        yield "Step 1/3: Loading documents..."
        loader = MultiDocumentLoader(file_paths)
        docs = loader.load()

        yield "Step 2/3: Splitting documents into hierarchical chunks..."
        splitter = HierarchicalSemanticSplitter(
            parent_chunk_size=self.config['splitter']['parent_chunk_size'],
            parent_chunk_overlap=self.config['splitter']['parent_chunk_overlap'],
            child_chunk_size=self.config['splitter']['child_chunk_size']
        )
        parent_docs, child_chunks = splitter.split_documents(docs)

        yield "Step 3/3: Building and saving vector index..."
        self.vector_store.build(parent_docs, child_chunks)
        self.is_ready = True
        yield "Knowledge base built and ready!"

    def _get_context_and_sources(self, query: str) ->  List[Document]:
        if not self.is_ready:
            raise Exception("Knowledge base is not ready. Please build it first.")

        # Hybrid Search to get child chunks
        retrieved_child_indices_scores = self.vector_store.search(
            query,
            top_k=self.config['retrieval']['retrieval_top_k'],
            alpha=self.config['retrieval']['hybrid_search_alpha']
        )
        retrieved_child_indices = [idx for idx, score in retrieved_child_indices_scores]
        retrieved_child_chunks = self.vector_store.get_chunks(retrieved_child_indices)

        # Get Parent Documents
        retrieved_parent_docs = self.vector_store.get_parent_docs(retrieved_child_chunks)

        # Rerank Parent Documents
        reranked_docs = self.llm.rerank(query, retrieved_parent_docs)
        final_context_docs = reranked_docs[:self.config['retrieval']['rerank_top_k']]

        return final_context_docs

    def get_response_full(self, query: str) ->(str, List[Document]):
        final_context_docs = self._get_context_and_sources(query)
        answer = self.llm.generate_answer(query, final_context_docs)
        return answer, final_context_docs

    def get_response_stream(self, query: str) ->(Generator[str, None, None], List[Document]):
        final_context_docs = self._get_context_and_sources(query)
        answer_generator = self.llm.generate_answer_stream(query, final_context_docs)
        return answer_generator, final_context_docs

    def get_context_string(self, context_docs: List[Document]) -> str:
        context_str = "引用上下文 (Context Sources):\n\n"
        for doc in context_docs:
            source_info = f"--- (来源: {os.path.basename(doc.metadata.get('source', ''))}, 页码: {doc.metadata.get('page', 'N/A')}) ---\n"
            content = doc.text[:200]+"..." if len(doc.text) > 200 else doc.text
            context_str += source_info + content + "\n\n"
        return context_str.strip()



