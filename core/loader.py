#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 19:51
# @Author  : hukangzhe
# @File    : loader.py
# @Description : 文档加载模块,不在像reg_mini返回长字符串，而是返回一个document对象列表，每个document都带有源文件信息
import logging
from .schema import Document
from typing import List
import fitz
import os


class DocumentLoader:

    @staticmethod
    def _load_pdf(file_path):
        logging.info(f"Loading PDF file from {file_path}")
        try:
            with fitz.open(file_path) as f:
                text = "".join(page.get_text() for page in f)
                logging.info(f"Successfully loaded {len(f)} pages.")
                return text
        except Exception as e:
            logging.error(f"Failed to load PDF {file_path}: {e}")
            return None


class MultiDocumentLoader:
    def __init__(self, paths: List[str]):
        self.paths = paths

    def load(self) -> List[Document]:
        docs = []
        for file_path in self.paths:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension == '.pdf':
                docs.extend(self._load_pdf(file_path))
            elif file_extension == '.txt':
                docs.extend(self._load_txt(file_path))
            else:
                logging.warning(f"Unsupported file type:{file_extension}. Skipping {file_path}")

        return docs

    def _load_pdf(self, file_path: str) -> List[Document]:
        logging.info(f"Loading PDF file from {file_path}")
        try:
            pdf_docs = []
            with fitz.open(file_path) as doc:
                for i, page in enumerate(doc):
                    pdf_docs.append(Document(
                        text=page.get_text(),
                        metadata={'source': file_path, 'page': i + 1}
                    ))
                return pdf_docs
        except Exception as e:
            logging.error(f"Failed to load PDF {file_path}: {e}")
            return []

    def _load_txt(self, file_path: str) -> List[Document]:
        logging.info(f"Loading txt file from {file_path}")
        try:
            txt_docs = []
            with open(file_path, 'r', encoding="utf-8") as f:
                txt_docs.append(Document(
                    text=f.read(),
                    metadata={'source': file_path}
                ))
                return txt_docs
        except Exception as e:
            logging.error(f"Failed to load txt {file_path}:{e}")
            return []
