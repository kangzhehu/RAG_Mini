#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/24 14:04
# @Author  : hukangzhe
# @File    : rag_core.py
# @Description :非常简单的RAG系统

import PyPDF2
import fitz
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import faiss
import torch


class RAGSystem:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.texts = self._load_and_spilt_pdf()
        self.embedder = SentenceTransformer('moka-ai/m3e-base')
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')  # 加载一个reranker模型
        self.vector_store = self._create_vector_store()
        print("3. Initializing Generator Model...")
        model_name = "Qwen/Qwen1.5-1.8B-Chat"

        # 检查是否有可用的GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   - Using device: {device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 注意：对于像Qwen这样的模型，我们通常使用 AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        self.generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=self.tokenizer,
            device=device
        )

    # 1. 文档加载 & 2.文本切分 （为了简化，合在一起）
    def _load_and_spilt_pdf(self):
        print("1. Loading and splitting PDF...")
        full_text = ""
        with fitz.open(self.pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()

        # 非常基础的切分：根据固定大小
        chunk_size = 500
        overlap = 50
        chunks = [full_text[i: i+chunk_size] for i in range(0, len(full_text), chunk_size-overlap)]
        print(f"   - Splitted into {len(chunks)} chunks.")
        return chunks

    # 3. 文本向量化 & 向量存储
    def _create_vector_store(self):
        print("2. Creating vector store...")
        # embedding
        embeddings = self.embedder.encode(self.texts)

        # Storing with faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)  # 使用L2距离进行相似度计算
        index.add(np.array(embeddings))
        print(" - Created vector store")
        return index

    # 4.检索
    def retrieve(self, query, k=3):
        print(f"3. Retrieving top {k} relevant chunks for query: '{query}' ")
        query_embeddings = self.embedder.encode([query])

        distances, indices = self.vector_store.search(np.array(query_embeddings), k=k)
        retrieved_chunks = [self.texts[i] for i in indices[0]]
        print("   - Retrieval complete.")
        return retrieved_chunks

    # 5.生成
    def generate(self, query, context_chunks):
        print("4. Generate answer...")
        context = "\n".join(context_chunks)

        messages = [
            {"role": "system", "content": "你是一个问答助手，请根据提供的上下文来回答问题，不要编造信息。"},
            {"role": "user", "content": f"上下文：\n---\n{context}\n---\n请根据以上上下文回答这个问题：{query}"}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # print("Final Prompt:\n", prompt)
        # print("Prompt token length:", len(self.tokenizer.encode(prompt)))

        result = self.generator(prompt, max_new_tokens=200, num_return_sequences=1,
                                eos_token_id=self.tokenizer.eos_token_id)

        print("   - Generation complete.")
        # print("Raw results:", result)
        # 提取生成的文本
        # 注意：Qwen模型返回的文本包含了prompt，我们需要从中提取出答案部分
        full_response = result[0]["generated_text"]
        answer = full_response[len(prompt):].strip()  # 从prompt之后开始截取

        # print("Final Answer:", repr(answer))
        return answer

    # 优化1
    def rerank(self, query, chunks):
        print("   - Reranking retrieved chunks...")
        pairs = [[query, chunk] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        # 将chunks和scores打包，并按score降序排序
        ranked_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in ranked_chunks]

    def query(self, query_text):
        # 1. 检索（可以检索更多结果，如top 10）
        retrieved_chunks = self.retrieve(query_text, k=10)

        # 2. 重排（从10个中选出最相关的3个）
        reranked_chunks = self.rerank(query_text, retrieved_chunks)
        top_k_reranked = reranked_chunks[:3]

        answer = self.generate(query_text, top_k_reranked)
        return answer


def main():
    # 确保你的data文件夹里有一个叫做sample.pdf的文件
    pdf_path = 'data/chinese_document.pdf'

    print("Initializing RAG System...")
    rag_system = RAGSystem(pdf_path)
    print("\nRAG System is ready. You can start asking questions.")
    print("Type 'q' to quit.")

    while True:
        user_query = input("\nYour Question: ")
        if user_query.lower() == 'q':
            break

        answer = rag_system.query(user_query)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()