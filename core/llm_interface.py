#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/29 19:54
# @Author  : hukangzhe
# @File    : generator.py
# @Description : 负责生成答案模块
import os
import queue
import logging
import threading

import torch
from typing import Dict, List, Tuple, Generator
from sentence_transformers import CrossEncoder
from .schema import Document, Chunk
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, TextStreamer

class ThinkStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool =True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.is_thinking = True
        self.think_end_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        self.output_queue = queue.Queue()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.output_queue.put(text)
        if stream_end:
            self.output_queue.put(None) # 发送结束信号

    def __iter__(self):
        return self

    def __next__(self):
        value = self.output_queue.get()
        if value is None:
            raise StopIteration()
        return value

    def generate_output(self) -> Generator[Tuple[str, str], None, None]:
        """
        分离Think和回答
        :return:
        """
        full_decode_text = ""
        already_yielded_len = 0
        for text_chunk in self:
            if not self.is_thinking:
                yield "answer", text_chunk
                continue

            full_decode_text += text_chunk
            tokens = self.tokenizer.encode(full_decode_text, add_special_tokens=False)

            if self.think_end_token_id in tokens:
                spilt_point = tokens.index(self.think_end_token_id)
                think_part_tokens = tokens[:spilt_point]
                thinking_text = self.tokenizer.decode(think_part_tokens)

                answer_part_tokens = tokens[spilt_point:]
                answer_text = self.tokenizer.decode(answer_part_tokens)
                remaining_thinking = thinking_text[already_yielded_len:]
                if remaining_thinking:
                    yield "thinking", remaining_thinking

                if answer_text:
                    yield "answer", answer_text

                self.is_thinking = False
                already_yielded_len = len(thinking_text) + len(self.tokenizer.decode(self.think_end_token_id))
            else:
                yield "thinking", text_chunk
                already_yielded_len += len(text_chunk)


class QueueTextStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.output_queue = queue.Queue()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Puts text into the queue; sends None as a sentinel value to signal the end."""
        self.output_queue.put(text)
        if stream_end:
            self.output_queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.output_queue.get()
        if value is None:
            raise StopIteration()
        return value


class LLMInterface:
    def __init__(self, config: dict):
        self.config = config
        self.reranker = CrossEncoder(config['models']['reranker'])
        self.generator_new_tokens = config['generation']['max_new_tokens']
        self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generator_name = config['models']['llm_generator']
        logging.info(f"Initializing generator {generator_name}")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_name,
            torch_dtype="auto",
            device_map="auto")

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        pairs = [[query, doc.text] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs]

    def _threaded_generate(self, streamer: QueueTextStreamer, generation_kwargs: dict):
        """
        一个包装函数，将 model.generate 放入 try...finally 块中。
        """
        try:
            self.generator_model.generate(**generation_kwargs)
        finally:
            # 无论成功还是失败，都确保在最后发送结束信号
            streamer.output_queue.put(None)

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        context_str = ""
        for doc in context_docs:
            context_str += f"Source: {os.path.basename(doc.metadata.get('source', ''))}, Page: {doc.metadata.get('page', 'N/A')}\n"
            context_str += f"Content: {doc.text}\n\n"
        # content设置为英文，回答则为英文
        messages = [
            {"role": "system", "content": "你是一个问答助手，请根据提供的上下文来回答问题，不要编造信息。"},
            {"role": "user", "content": f"上下文：\n---\n{context_str}\n---\n请根据以上上下文回答这个问题：{query}"}
        ]
        prompt = self.generator_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.generator_model.generate(**inputs,
                                               max_new_tokens=self.generator_new_tokens, num_return_sequences=1,
                                eos_token_id=self.generator_tokenizer.eos_token_id)
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        answer = self.generator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer

    def generate_answer_stream(self, query: str, context_docs: List[Document]) -> Generator[str, None, None]:
        context_str = ""
        for doc in context_docs:
            context_str += f"Content: {doc.text}\n\n"

        messages = [
            {"role": "system",
             "content": "你是一个问答助手，请根据提供的上下文来回答问题，不要编造信息。"},
            {"role": "user",
             "content": f"上下文:\n---\n{context_str}\n---\n请根据以上上下文回答这个问题： {query}"}
        ]

        prompt = self.generator_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.generator_tokenizer([prompt], return_tensors="pt").to(self.device)

        streamer = QueueTextStreamer(self.generator_tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=self.generator_new_tokens,
            streamer=streamer,
            pad_token_id=self.generator_tokenizer.eos_token_id,
        )

        thread = threading.Thread(target=self._threaded_generate, args=(streamer,generation_kwargs,))
        thread.start()
        for new_text in streamer:
            if new_text is not None:
                yield new_text

    def generate_answer_stream_split(self, query: str, context_docs: List[Document]) -> Generator[Tuple[str, str], None, None]:
        """分离思考和回答的流式输出"""
        context_str = ""
        for doc in context_docs:
            context_str += f"Content: {doc.text}\n\n"

        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Please answer the question based on the provided context. First, think through the process in <think> tags, then provide the final answer."},
            {"role": "user",
             "content": f"Context:\n---\n{context_str}\n---\nBased on the context above, please answer the question: {query}"}
        ]

        prompt = self.generator_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.generator_tokenizer([prompt], return_tensors="pt").to(self.device)

        streamer = ThinkStreamer(self.generator_tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=self.generator_new_tokens,
            streamer=streamer
        )

        thread = threading.Thread(target=self.generator_model.generate, kwargs=generation_kwargs)
        thread.start()

        yield from streamer.generate_output()




