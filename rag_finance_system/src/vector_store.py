import chromadb
from loguru import logger
from typing import List, Dict, Any, Optional
import uuid

COLLECTION_NAME = "finance_regulations"

class VectorStore:
    def __init__(self):
        logger.info("初始化 Chroma 向量库")

        self.client = chromadb.Client(
            settings=chromadb.config.Settings(
                persist_directory="./db/chroma"
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME
        )

    
    def insert(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 4000,   # 👈 新增参数
    ) -> int:

        assert len(chunks) == len(embeddings), "chunks与embeddings数量不匹配"

        total = len(chunks)
        inserted = 0

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [str(uuid.uuid4()) for _ in batch_chunks]

            documents = [chunk["text"][:4096] for chunk in batch_chunks]

            metadatas = []
            for chunk in batch_chunks:
                metadatas.append({
                    "source": chunk.get("source", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "article_num": chunk.get("article_num", ""),
                    "file_path": chunk.get("file_path", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "doc_type": chunk.get("doc_type", "law"),
                    "law_name": chunk.get("law_name", ""),
                    "authority": chunk.get("authority", ""),
                })

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )

            inserted += len(batch_chunks)
            logger.info(f"已插入 {inserted}/{total}")

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

        where = {}
        if source_filter:
            where["source"] = source_filter
        if doc_type_filter:
            where["doc_type"] = doc_type_filter
        if law_name_filter:
            where["law_name"] = law_name_filter
        if authority_filter:
            where["authority"] = authority_filter
        if not where:
            where = None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "score": 1 - results["distances"][0][i] / 4,
                "text": results["documents"][0][i],
                **results["metadatas"][0][i],
            })

        return hits

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "count": self.collection.count()
        }

    def drop_collection(self):
        self.client.delete_collection(COLLECTION_NAME)
        logger.warning(f"Collection [{COLLECTION_NAME}] 已删除")