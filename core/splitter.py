#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/25 19:52
# @Author  : hukangzhe
# @File    : splitter.py
# @Description : 负责切分文本的模块

import logging
from typing import List, Dict, Tuple
from .schema import Document, Chunk


class SemanticRecursiveSplitter:
    def __init__(self, chunk_size: int=500, chunk_overlap:int = 50, separators: List[str] = None):
        """
        一个真正实现递归的语义文本切分器。
        :param chunk_size: 每个文本块的目标大小。
        :param chunk_overlap: 文本块之间的重叠大小。
        :param separators: 用于切分的语义分隔符列表，按优先级从高到低排列。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_size <= self.chunk_overlap:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self.separators = separators if separators else ['\n\n', '\n', " ", ""] # 默认分割符

    def text_split(self, text: str) -> List[str]:
        """
        切分入口
        :param text:
        :return:
        """
        logging.info("Starting semantic recursive splitting...")
        final_chunks = self._split(text, self.separators)
        logging.info(f"Text successfully split into {len(final_chunks)} chunks.")
        return final_chunks

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        # 1. 如果文本足够小，直接返回
        if len(text) < self.chunk_size:
            return [text]
        # 2. 先尝试最高优先的分割符
        cur_separator = separators[0]

        # 3. 如果可以分割
        if cur_separator in text:
            # 分割成多个小部分
            parts = text.split(cur_separator)

            buffer=""   # 用来合并小部分
            for i, part in enumerate(parts):
                # 如果小于chunk_size,就再加一小部分，使buffer接近chunk_size
                if len(buffer) + len(part) + len(cur_separator) <= self.chunk_size:
                    buffer += part+cur_separator
                else:
                    # 如果buffer 不为空
                    if buffer:
                        final_chunks.append(buffer)
                    # 如果当前part就已经超过chunk_size
                    if len(part) > self.chunk_size:
                        # 递归调用下一级
                        sub_chunks = self._split(part, separators = separators[1:])
                        final_chunks.extend(sub_chunks)
                    else:   # 成为新的缓冲区
                        buffer = part + cur_separator

            if buffer:  # 最后一部分的缓冲区
                final_chunks.append(buffer.strip())

        else:
            # 4. 使用下一级分隔符
            final_chunks = self._split(text, separators[1:])

        # 处理重叠
        if self.chunk_overlap > 0:
            return self._handle_overlap(final_chunks)
        else:
            return final_chunks

    def _handle_overlap(self, final_chunks: List[str]) -> List[str]:
        overlap_chunks = []
        if not final_chunks:
            return []
        overlap_chunks.append(final_chunks[0])
        for i in range(1, len(final_chunks)):
            pre_chunk = overlap_chunks[-1]
            cur_chunk = final_chunks[i]
            # 从前一个chunk取出重叠部分与当前chunk合并
            overlap_part = pre_chunk[-self.chunk_overlap:]
            overlap_chunks.append(overlap_part+cur_chunk)

        return overlap_chunks


class HierarchicalSemanticSplitter:
    """
    结合了层次化（父/子）和递归语义分割策略。确保在创建父块和子块时，遵循文本的自然语义边界。
    """
    def __init__(self,
                 parent_chunk_size: int = 800,
                 parent_chunk_overlap: int = 100,
                 child_chunk_size: int = 250,
                 separators: List[str] = None):
        if parent_chunk_overlap >= parent_chunk_size:
            raise ValueError("Parent chunk overlap must be smaller than parent chunk size.")
        if child_chunk_size >= parent_chunk_size:
            raise ValueError("Child chunk size must be smaller than parent chunk size.")

        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.separators = separators or ["\n\n", "\n", "。", ". ", "！", "!", "？", "?", " ", ""]

    def _recursive_semantic_split(self, text: str, chunk_size: int) -> List[str]:
        """
        优先考虑语义边界
        """
        if len(text) <= chunk_size:
            return [text]

        for sep in self.separators:
            split_point = text.rfind(sep, 0, chunk_size)
            if split_point != -1:
                break
        else:
            split_point = chunk_size

        chunk1 = text[:split_point]
        remaining_text = text[split_point:].lstrip()  # 删除剩余部分的前空格

        # 递归拆分剩余文本
        # 分隔符将添加回第一个块以保持上下文
        if remaining_text:
            return [chunk1 + (sep if sep in " \n" else "")] + self._recursive_semantic_split(remaining_text, chunk_size)
        else:
            return [chunk1]

    def _apply_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """处理重叠部分chunk"""
        if not overlap or len(chunks) <= 1:
            return chunks

        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            # 从前一个chunk中获取最后的“重叠”字符
            overlap_content = chunks[i - 1][-overlap:]
            overlapped_chunks.append(overlap_content + chunks[i])

        return overlapped_chunks

    def split_documents(self, documents: List[Document]) -> Tuple[Dict[int, Document], List[Chunk]]:
        """
        两次切分
        :param documents:
        :return:
            -  parent documents: {parent_id: Document}
            -  child chunks: [Chunk, Chunk, ...]
        """
        parent_docs_dict: Dict[int, Document] = {}
        child_chunks_list: List[Chunk] = []
        parent_id_counter = 0

        logging.info("Starting robust hierarchical semantic splitting...")

        for doc in documents:
            # === PASS 1: 创建父chunks ===
            # 1. 将整个文档text分割成大的语义chunks
            initial_parent_chunks = self._recursive_semantic_split(doc.text, self.parent_chunk_size)

            # 2. 父chunks进行重叠处理
            overlapped_parent_texts = self._apply_overlap(initial_parent_chunks, self.parent_chunk_overlap)

            for p_text in overlapped_parent_texts:
                parent_doc = Document(text=p_text, metadata=doc.metadata.copy())
                parent_docs_dict[parent_id_counter] = parent_doc

                # === PASS 2: Create Child Chunks from each Parent ===
                child_texts = self._recursive_semantic_split(p_text, self.child_chunk_size)

                for c_text in child_texts:
                    child_metadata = doc.metadata.copy()
                    child_metadata['parent_id'] = parent_id_counter
                    child_chunk = Chunk(text=c_text, metadata=child_metadata, parent_id=parent_id_counter)
                    child_chunks_list.append(child_chunk)

                parent_id_counter += 1

        logging.info(
            f"Splitting complete. Created {len(parent_docs_dict)} parent chunks and {len(child_chunks_list)} child chunks.")
        return parent_docs_dict, child_chunks_list
