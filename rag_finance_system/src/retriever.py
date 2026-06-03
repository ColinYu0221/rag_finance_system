"""
retriever.py
检索模块：向量检索 + Reranker精排
Phase 1实现纯向量检索，Phase 2扩展混合检索
通用检索器
"""

import os
from typing import List, Dict, Any, Optional

from loguru import logger
from dotenv import load_dotenv

from .embedder import Embedder, Reranker
from .vector_store import VectorStore

load_dotenv()

TOP_K = int(os.getenv("RETRIEVER_TOP_K", 10))
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", 5))


class Retriever:
    """
    检索器
    Phase 1：向量检索 → Reranker精排
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        reranker: Optional[Reranker] = None,
        top_k: int = TOP_K,
        reranker_top_n: int = RERANKER_TOP_N,
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.reranker = reranker  # 可选，不传则跳过精排
        self.top_k = top_k
        self.reranker_top_n = reranker_top_n

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        law_name_filter: Optional[str] = None,
        authority_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        端到端检索：query → 相关chunks
        Args:
            query: 用户问题
            top_k: 向量检索数量（覆盖默认值）
            use_reranker: 是否启用Reranker精排
            source_filter: 按文档来源过滤
            doc_type_filter: 按文档类型过滤（law/case）
            law_name_filter: 按法律名称精确过滤（如"中华人民共和国公司法"）
            authority_filter: 按发布机构过滤（如"上海银保监局"）
        Returns:
            排序后的chunk列表，每条包含text、source、score等
        """
        k = top_k or self.top_k

        # Step 1: 查询Embedding
        logger.info(f"检索: {query[:50]}...")
        query_vector = self.embedder.encode_query(query)

        # Step 2: 向量检索（多内容过滤器拆为 OR 查询，doc_type 始终 AND）
        # 法律名、发布机构、来源文件 属于不同元数据维度，不可能同时命中同一条 chunk，
        # 拆分成多次独立查询后合并去重。
        # 发布机构可能为逗号分隔的多值（如 "上海保监局,上海银保监局"）
        content_filters: list[dict] = []
        if law_name_filter:
            content_filters.append({"law_name": law_name_filter})
        if authority_filter:
            for _auth in authority_filter.split(","):
                _auth = _auth.strip()
                if _auth:
                    content_filters.append({"authority": _auth})
        if source_filter:
            content_filters.append({"source": source_filter})
        if not content_filters:
            content_filters.append({})  # 无内容过滤，跑一次全量

        all_candidates: dict[str, dict] = {}  # id → candidate
        for cf in content_filters:
            # 每次查询都叠加 doc_type_filter（结构类型，始终 AND）
            where = {**cf}
            if doc_type_filter:
                where["doc_type"] = doc_type_filter

            batch = self.vector_store.search(
                query_vector=query_vector,
                top_k=k,
                source_filter=where.get("source"),       # None 即不过滤
                doc_type_filter=where.get("doc_type"),
                law_name_filter=where.get("law_name"),
                authority_filter=where.get("authority"),
            )

            for c in batch:
                cid = c.get("id", "")
                if cid and cid not in all_candidates:
                    all_candidates[cid] = c

        # 按分数排序，取 top_k
        candidates = sorted(all_candidates.values(),
                            key=lambda x: x.get("score", 0.0), reverse=True)[:k]
        logger.info(f"向量检索召回 {len(candidates)} 条 "
                    f"(content_filters={len(content_filters)})")

        if not candidates:
            return []

        # Step 3: Reranker精排（可选）
        if use_reranker and self.reranker:
            texts = [c["text"] for c in candidates]
            reranked = self.reranker.rerank(
                query=query,
                documents=texts,
                top_n=self.reranker_top_n,
            )
            # 将reranker结果与原始候选合并
            results = []
            for r in reranked:
                orig = candidates[r["index"]]
                orig["reranker_score"] = r["score"]
                results.append(orig)
            logger.info(f"Reranker精排后保留 {len(results)} 条")
            return results
        else:
            return candidates[:self.reranker_top_n]

    def compute_confidence(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        计算答案可信度（创新点2的基础版本）
        Returns: {"total": float, "retrieval": float, "coverage": float}
        """
        if not retrieved_chunks:
            return {"total": 0.0, "retrieval": 0.0, "coverage": 0.0}

        # 检索相关性：取最高向量分数（已在 [0, 1] 范围）
        retrieval_score = max(c.get("score", 0.0) for c in retrieved_chunks)

        # 答案覆盖度：检查答案中有多少内容来自检索片段
        answer_lower = answer.lower()
        matched = sum(
            1 for c in retrieved_chunks
            if any(word in answer_lower for word in c["text"][:100].lower().split()[:10])
        )
        coverage_score = min(matched / max(len(retrieved_chunks), 1), 1.0)

        total = 0.6 * retrieval_score + 0.4 * coverage_score

        return {
            "total": round(total, 3),
            "retrieval": round(retrieval_score, 3),
            "coverage": round(coverage_score, 3),
        }
