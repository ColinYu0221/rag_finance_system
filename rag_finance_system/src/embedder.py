"""
embedder.py
Embedding模型封装（bge-small-zh-v1.5）
支持：文本→向量，批量编码，GPU自动检测
"""

import os
from typing import List, Union
import torch.nn.functional as F
import torch
from loguru import logger
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-small-zh-v1.5")
RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", 16))


class Embedder:
    """
    Embedding模型封装
    模型：bge-small-zh-v1.5，输出维度由模型配置决定
    """

    def __init__(self, model_path: str = EMBEDDING_MODEL_PATH):
        from sentence_transformers import SentenceTransformer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载Embedding模型: {model_path}，设备: {self.device}")
        self.model = SentenceTransformer(model_path, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding维度: {self.dimension}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        文本→向量
        Args:
            texts: 单条文本或文本列表
            batch_size: 批处理大小（显存不足时调小）
            show_progress: 显示进度条
        Returns:
            向量列表，每个向量为float列表
        """
        if isinstance(texts, str):
            texts = [texts]

        # bge模型对query加前缀效果更好
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2归一化，便于余弦相似度计算
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def encode_query(self, query: str) -> List[float]:
        """
        查询编码（加Instruction前缀，提升检索效果）
        """
        # bge中文模型推荐的query前缀
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        result = self.encode(prefixed)
        return result[0]

    def encode_documents(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """文档批量编码（不加前缀）"""
        return self.encode(texts, batch_size=batch_size, show_progress=True)



class Reranker:
    def __init__(self, model_path: str = RERANKER_MODEL_PATH, batch_size: int = RERANKER_BATCH_SIZE):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        load_kwargs = {}
        if self.device == "cuda":
            load_kwargs["dtype"] = torch.float16

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            **load_kwargs
        ).to(self.device)
        self.model.eval()

        if self.device == "cuda":
            self._warmup()

    def _warmup(self):
        dummy_pairs = [["warmup", "warmup"]]
        inputs = self.tokenizer(
            dummy_pairs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            _ = self.model(**inputs)

    def rerank(self, query, documents, top_n=None):
        if not documents:
            return []

        all_scores = []
        all_indices = []

        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(logits)

            scores = scores.cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)
            all_indices.extend(range(i, i + len(batch_docs)))

        results = [
            {"index": idx, "score": score}
            for idx, score in zip(all_indices, all_scores)
        ]
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        if top_n:
            results = results[:top_n]

        return results