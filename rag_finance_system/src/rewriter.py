"""
rewriter.py
查询重写专用轻量本地模型。
默认 Qwen2.5-0.5B-Instruct，CPU 推理 <100ms，支持 LoRA 加载微调权重。
"""
import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger
from dotenv import load_dotenv

load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

# 确保 HuggingFace 镜像生效
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

_REWRITER_MODEL_PATH = os.getenv("REWRITER_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")

# 相对路径 → src/ 向上两级 (项目根) 再拼 models/
_mp = Path(_REWRITER_MODEL_PATH)
if not _mp.is_absolute():
    _mp = (Path(__file__).resolve().parent.parent.parent / _REWRITER_MODEL_PATH).resolve()
REWRITER_MODEL_PATH = str(_mp)

REWRITER_LORA_PATH = os.getenv("REWRITER_LORA_PATH", "")

QUERY_REWRITE_PROMPT = """你是一个中文检索查询改写助手。
你的任务是将用户的自然语言问题改写为一条简洁、精准、适合向量检索的查询语句。
请保留关键实体和意图，去掉无关废话，仅输出改写后的查询，不要输出额外解释或格式说明。"""


class QueryRewriter:
    """
    查询重写小模型。

    可调参数:
        model_path  : 模型路径或 HuggingFace ID
        lora_path   : 微调后的 LoRA 适配器路径（空则不加载）
        load_in_4bit: GPU 上启用 INT4 量化
        device      : 指定设备，默认自动检测
    """

    def __init__(
        self,
        model_path: str = REWRITER_MODEL_PATH,
        lora_path: str = REWRITER_LORA_PATH,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"加载查询重写模型: {model_path}，设备: {self.device}")

        try:
            if load_in_4bit and self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    load_in_4bit=True,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                if self.device == "cpu":
                    self.model = self.model.to(self.device)

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

            if lora_path and os.path.exists(lora_path):
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, lora_path)
                logger.info(f"已加载 LoRA 适配器: {lora_path}")

            self.model.eval()
            self._available = True
            logger.info("查询重写模型加载成功")
        except Exception as e:
            self._available = False
            logger.warning(f"查询重写模型本地不可用，将跳过查询重写: {e}")

    def rewrite(
        self,
        question: str,
        temperature: float = 0.1,
        max_new_tokens: int = 64,
    ) -> str:
        """将自然语言问题改写为检索查询。"""
        if not getattr(self, "_available", False):
            return question
        messages = [
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            {
                "role": "user",
                "content": (
                    "请将以下问题改写为适合向量检索的简洁查询：\n\n"
                    f"{question}\n\n只输出改写后的查询。"
                ),
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                repetition_penalty=1.1,
            )

        generated = output_ids[0][inputs.input_ids.shape[1] :]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return result or question
