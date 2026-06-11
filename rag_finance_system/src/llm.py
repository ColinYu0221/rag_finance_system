"""
llm.py
LLM推理模块
支持：本地Qwen2.5-7B-Int4（优先）、通义千问API（备选）
"""

import os
from typing import List, Optional, Generator

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "./models/Qwen2.5-7B-Int4")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_BASE_URL = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")


# ========================
# 本地推理（GPTQ量化模型）
# ========================

class LocalLLM:
    """
    本地Qwen2.5-7B-Instruct-GPTQ-Int4推理
    显存需求：~4.5GB（4060 8GB够用）
    """

    def __init__(self, model_path: str = LLM_MODEL_PATH):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载本地LLM: {model_path}，设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",           # 自动分配GPU/CPU
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("本地LLM加载完成")

    def _prepare_inputs(self, messages: List[dict]):
        import torch
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return self.tokenizer([text], return_tensors="pt").to(self.device)

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        import torch
        inputs = self._prepare_inputs(messages)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                do_sample=temperature > 0, repetition_penalty=1.1,
            )
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_stream(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ):
        """流式生成，yield 每个新 token 的文本增量。"""
        import torch
        from transformers import TextStreamer, TextIteratorStreamer
        from threading import Thread

        inputs = self._prepare_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            repetition_penalty=1.1,
            streamer=streamer,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        yield from streamer


# ========================
# API调用（通义千问，备选）
# ========================

class QwenAPILLM:
    """
    通义千问API调用（当本地显存不足时使用）
    免费额度：qwen-plus 每分钟1M tokens
    """

    def __init__(self, api_key: str = DASHSCOPE_API_KEY, model: str = "qwen-plus"):
        if not api_key:
            raise ValueError("请在.env中设置DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.model = model
        logger.info(f"使用通义千问API: {model}")

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


class DeepseekAPILLM:
    """
    Deepseek API调用（OpenAI兼容接口）
    """

    def __init__(self, api_key: str = DEEPSEEK_API_KEY, model: str = DEEPSEEK_MODEL, base_url: str = DEEPSEEK_API_BASE_URL):
        if not api_key:
            raise ValueError("请在.env中设置DEEPSEEK_API_KEY")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        logger.info(f"使用Deepseek API: {model}，base_url={base_url}")

    def generate(
        self,
        messages: List[dict],
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content


# ========================
# 工厂函数：自动选择LLM
# ========================

def get_llm(prefer_local: bool = True):
    """
    自动选择LLM：优先本地，失败则退回API
    """
    if prefer_local:
        try:
            return LocalLLM()
        except Exception as e:
            logger.warning(f"本地LLM加载失败: {e}，切换到API模式")

    if DEEPSEEK_API_KEY:
        return DeepseekAPILLM()
    if DASHSCOPE_API_KEY:
        return QwenAPILLM()

    raise RuntimeError(
        "无法初始化LLM：本地模型加载失败且未配置 Deepseek 或 通义千问 API Key。\n"
        "请在.env中设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY，或确保本地模型路径正确。"
    )
