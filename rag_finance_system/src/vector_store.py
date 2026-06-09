import os
import time
import uuid
from pathlib import Path as _StdPath
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from loguru import logger
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)
from pymilvus.milvus_client.index import IndexParams

# src/vector_store.py → rag_finance_system/.env
# 注意：不要设置 MILVUS_URI 环境变量，pymilvus 导入时会自动解析它。
# 使用 MILVUS_LOCAL_DB 作为本地 .db 文件路径（仅 MilvusClient 读取）。
load_dotenv(dotenv_path=str(_StdPath(__file__).resolve().parent.parent / ".env"))

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "finance_regulations")
DEFAULT_TEXT_MAX_LENGTH = 4096


class VectorStore:
    def __init__(self):
        self.collection_name = COLLECTION_NAME
        local_db = os.getenv("MILVUS_LOCAL_DB", "")
        host = os.getenv("MILVUS_HOST", "127.0.0.1")
        port = os.getenv("MILVUS_PORT", "19530")
        user = os.getenv("MILVUS_USER", "")
        password = os.getenv("MILVUS_PASSWORD", "")
        db_name = os.getenv("MILVUS_DB_NAME", "")
        self.embed_dim = self._read_embed_dim()
        self._index_type: Optional[str] = None

        # .db 本地文件 → POSIX 绝对路径；否则用 HTTP 连接
        if local_db:
            uri = _StdPath(local_db).resolve().as_posix()
        else:
            uri = f"http://{host}:{port}"

        self.client = self._connect(uri, user, password, db_name)
        self._ensure_loaded()

    def _read_embed_dim(self) -> Optional[int]:
        raw = os.getenv("MILVUS_EMBED_DIM", "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            logger.warning(f"MILVUS_EMBED_DIM 不是有效整数: {raw}，将改为首次写入时自动推断")
            return None

    def _connect(self, uri: str, user: str, password: str, db_name: str) -> MilvusClient:
        logger.info("初始化 Milvus 向量库")
        kwargs = {"uri": uri}
        if user:
            kwargs["user"] = user
        if password:
            kwargs["password"] = password
        if db_name:
            kwargs["db_name"] = db_name

        try:
            client = MilvusClient(**kwargs)
            return client
        except Exception as e:
            raise RuntimeError(
                "Milvus 连接失败，请检查 MILVUS_URI 或 MILVUS_HOST/MILVUS_PORT 配置。"
            ) from e

    def _ensure_loaded(self):
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            self._index_type = self._detect_index_type()
            logger.info(f"已加载 Milvus collection: {self.collection_name}")
        else:
            logger.info(f"Milvus collection 不存在，将在首次写入时创建: {self.collection_name}")

    def _detect_index_type(self) -> Optional[str]:
        try:
            indexes = self.client.list_indexes(self.collection_name)
            if indexes:
                idx_info = self.client.describe_index(self.collection_name, indexes[0])
                idx_type = idx_info.get("index_type", "")
                if idx_type:
                    return idx_type
        except Exception:
            pass

        try:
            desc = self.client.describe_collection(self.collection_name)
            for idx in desc.get("indexes", []):
                idx_type = idx.get("index_type", "")
                if idx_type:
                    return idx_type
            for field in desc.get("fields", []):
                if field.get("name") == "embedding":
                    for idx in (field.get("index_params") or []):
                        idx_type = idx.get("index_type", "")
                        if idx_type:
                            return idx_type
        except Exception:
            pass
        return None

    def _ensure_collection(self, dim: int):
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            self._index_type = self._detect_index_type()
            return

        self.embed_dim = self.embed_dim or dim
        if self.embed_dim != dim:
            raise ValueError(f"向量维度不匹配：collection={self.embed_dim}, 当前={dim}")

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embed_dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=DEFAULT_TEXT_MAX_LENGTH),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="article_num", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="law_name", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="authority", dtype=DataType.VARCHAR, max_length=512),
        ]
        schema = CollectionSchema(fields=fields, description="金融法规向量库")

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )
        self._ensure_index()
        self.client.load_collection(self.collection_name)
        logger.info(f"已创建 Milvus collection: {self.collection_name} (dim={self.embed_dim})")

    def _ensure_index(self):
        try:
            idxp = IndexParams()
            idxp.add_index(
                field_name="embedding",
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=idxp,
            )
            self._index_type = "AUTOINDEX"
            logger.info("Milvus 向量索引已创建: AUTOINDEX/COSINE")
        except Exception as autoindex_error:
            logger.warning(f"AUTOINDEX 创建失败，回退 HNSW: {autoindex_error}")
            idxp = IndexParams()
            idxp.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=idxp,
            )
            self._index_type = "HNSW"
            logger.info("Milvus 向量索引已创建: HNSW/COSINE")

    def _get_search_params(self) -> dict:
        if self._index_type and self._index_type.upper() == "HNSW":
            return {"metric_type": "COSINE", "params": {"ef": 64}}
        return {"metric_type": "COSINE", "params": {}}

    @staticmethod
    def _clean_text(value: Any, limit: int) -> str:
        return str(value or "")[:limit]

    @staticmethod
    def _escape_expr_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _build_expr(
        self,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        law_name_filter: Optional[str] = None,
        authority_filter: Optional[str] = None,
    ) -> Optional[str]:
        clauses = []
        for key, value in {
            "source": source_filter,
            "doc_type": doc_type_filter,
            "law_name": law_name_filter,
            "authority": authority_filter,
        }.items():
            if value:
                clauses.append(f'{key} == "{self._escape_expr_value(str(value))}"')
        return " and ".join(clauses) if clauses else ""

    @staticmethod
    def _normalize_score(raw_score: float) -> float:
        if 0.0 <= raw_score <= 1.0:
            return round(raw_score, 4)
        if -1.0 <= raw_score <= 1.0:
            return round((raw_score + 1.0) / 2.0, 4)
        return round(max(0.0, min(1.0, 1.0 - raw_score / 2.0)), 4)

    @staticmethod
    def _retry_operation(op, *args, max_retries: int = 5, operation_name: str = ""):
        """带退避的重试，处理 milvus-lite 在 Windows 下的文件锁冲突。"""
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                return op(*args)
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    wait = 0.5 * (2 ** attempt)  # 1s, 2s, 4s, 8s
                    logger.warning(f"Milvus {operation_name} 失败 (attempt {attempt}/{max_retries})，"
                                   f"{wait:.1f}s 后重试: {e}")
                    time.sleep(wait)
        raise last_exc

    def _insert_batch(self, data: list):
        """带重试的单批插入。"""
        return self._retry_operation(
            self.client.insert,
            collection_name=self.collection_name,
            data=data,
            operation_name="insert",
        )

    def _flush_with_retry(self):
        """带重试的 flush。"""
        return self._retry_operation(
            self.client.flush,
            self.collection_name,
            operation_name="flush",
        )

    def insert(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 4000,
    ) -> int:
        assert len(chunks) == len(embeddings), "chunks与embeddings数量不匹配"
        if not chunks:
            return 0

        self._ensure_collection(len(embeddings[0]))

        total = len(chunks)
        inserted = 0

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            data = []
            for chunk, emb in zip(batch_chunks, batch_embeddings):
                data.append({
                    "id": str(uuid.uuid4()),
                    "embedding": emb,
                    "text": self._clean_text(chunk.get("text", ""), DEFAULT_TEXT_MAX_LENGTH),
                    "source": self._clean_text(chunk.get("source", ""), 512),
                    "chunk_id": self._clean_text(chunk.get("chunk_id", ""), 512),
                    "article_num": self._clean_text(chunk.get("article_num", ""), 64),
                    "file_path": self._clean_text(chunk.get("file_path", ""), 2048),
                    "chunk_index": int(chunk.get("chunk_index", 0)),
                    "doc_type": self._clean_text(chunk.get("doc_type", "law"), 64),
                    "law_name": self._clean_text(chunk.get("law_name", ""), 512),
                    "authority": self._clean_text(chunk.get("authority", ""), 512),
                })

            self._insert_batch(data=data)
            inserted += len(batch_chunks)
            logger.info(f"已插入 {inserted}/{total}")

        self._flush_with_retry()
        self.client.load_collection(self.collection_name)
        return inserted

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        law_name_filter: Optional[str] = None,
        authority_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client.has_collection(self.collection_name):
            return []

        self.client.load_collection(self.collection_name)

        expr = self._build_expr(
            source_filter=source_filter,
            doc_type_filter=doc_type_filter,
            law_name_filter=law_name_filter,
            authority_filter=authority_filter,
        )

        search_params = self._get_search_params()
        output_fields = [
            "text",
            "source",
            "chunk_id",
            "article_num",
            "file_path",
            "chunk_index",
            "doc_type",
            "law_name",
            "authority",
        ]

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            filter=expr if expr else "",
            limit=top_k,
            output_fields=output_fields,
            search_params=search_params,
            anns_field="embedding",
        )

        hits = []
        for result_list in results:
            for hit in result_list:
                entity = hit.get("entity", {})
                hits.append({
                    "id": str(hit.get("id", "")),
                    "score": self._normalize_score(float(hit.get("distance", 0.0))),
                    "text": entity.get("text", ""),
                    "source": entity.get("source", ""),
                    "chunk_id": entity.get("chunk_id", ""),
                    "article_num": entity.get("article_num", ""),
                    "file_path": entity.get("file_path", ""),
                    "chunk_index": entity.get("chunk_index", 0),
                    "doc_type": entity.get("doc_type", "law"),
                    "law_name": entity.get("law_name", ""),
                    "authority": entity.get("authority", ""),
                })

        return hits

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.client.has_collection(self.collection_name):
            return {"count": 0, "row_count": 0}

        stats = self.client.get_collection_stats(self.collection_name)
        count = stats.get("row_count", 0)
        return {"count": count, "row_count": count}

    def drop_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            self._index_type = None
            logger.warning(f"Collection [{self.collection_name}] 已删除")