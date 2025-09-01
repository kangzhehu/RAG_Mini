#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/29 20:03
# @Author  : hukangzhe
# @File    : test_qw.py
# @Description : 测试两个模型(qwen1.5 qwen3)的两种输出方式(full or stream)是否正确
import os
import queue
import logging
import threading
import torch
from typing import Tuple, Generator
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



class LLMInterface:
    def __init__(self, model_name: str= "Qwen/Qwen3-0.6B"):
        logging.info(f"Initializing generator {model_name}")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def generate_answer(self, query: str, context_str: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个问答助手，请根据提供的上下文来回答问题，不要编造信息。"},
            {"role": "user", "content": f"上下文：\n---\n{context_str}\n---\n请根据以上上下文回答这个问题：{query}"}
        ]
        prompt = self.generator_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.generator_tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.generator_model.generate(**inputs,
                                               max_new_tokens=256, num_return_sequences=1,
                                eos_token_id=self.generator_tokenizer.eos_token_id)
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        answer = self.generator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return answer

    def generate_answer_stream(self, query: str, context_str: str) -> Generator[Tuple[str, str], None, None]:
        """Generates an answer as a stream of (state, content) tuples."""
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Please answer the question based on the provided context. First, think through the process in <think> tags, then provide the final answer."},
            {"role": "user",
             "content": f"Context:\n---\n{context_str}\n---\nBased on the context above, please answer the question: {query}"}
        ]

        # Use the template that enables thinking for Qwen models
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
            max_new_tokens=512,
            streamer=streamer
        )

        thread = threading.Thread(target=self.generator_model.generate, kwargs=generation_kwargs)
        thread.start()

        yield from streamer.generate_output()


# if __name__ == "__main__":
#     qwen = LLMInterface("Qwen/Qwen3-0.6B")
#     answer = qwen.generate_answer("儒家思想的创始人是谁？", "中国传统哲学以儒家、道家和法家为主要流派。儒家思想由孔子创立，强调“仁”、“义”、“礼”、“智”、“信”，主张修身齐家治国平天下，对中国社会产生了深远的影响。其核心价值观如“己所不欲，勿施于人”至今仍具有普世意义。"+
#
# "道家思想以老子和庄子为代表，主张“道法自然”，追求人与自然的和谐统一，强调无为而治、清静无为。道家思想对中国人的审美情趣、艺术创作以及养生之道都有着重要的影响。"+
#
# "法家思想以韩非子为集大成者，主张以法治国，强调君主的权威和法律的至高无上。尽管法家思想在历史上曾被用于强化中央集权，但其对建立健全的法律体系也提供了重要的理论基础。")
#
#     print(answer)
