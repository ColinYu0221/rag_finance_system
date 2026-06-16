"""
api_app.py
FastAPI 应用 — 从 Streamlit app.py 平移核心逻辑到 REST API。
启动: py -3 -m uvicorn rag_finance_system.api_app:app --host 0.0.0.0 --port 8000
"""

import json
import re
import sys
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

# api_app.py 在 rag_finance_system/ 下，同级 .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_finance_system.api_schemas import (
    ArticleRelationRequest,
    ArticleRelationResponse,
    CategoryOut,
    CategoryRename,
    ConfidenceScores,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    DictionaryCategoryUpdate,
    DictionaryItemOut,
    DictionaryItemsResponse,
    FavoriteCreate,
    FavoriteOut,
    FavoritesCheckOut,
    FlowchartImageRequest,
    FlowchartResponse,
    FlowchartTextRequest,
    IndexRequest,
    IndexResponse,
    LawNamesResponse,
    MessageOut,
    QARequest,
    QAResponse,
    QAStreamRequest,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    SourceItem,
    UploadResponse,
)

# ── 懒加载单例 ──
_embedder = None
_vector_store = None
_reranker = None
_reranker_failed = False
_bm25_index = None
_bm25_path = str(Path(__file__).resolve().parent / "data" / "bm25_index.pkl")
_dict_path = str(Path(__file__).resolve().parent.parent / "data" / "finance_dictionary.json")
_dictionary = None
_dictionary_failed = False
_rag = None
_processor = None
_llm = None

# ES 单例
_es_index = None
_es_failed = False
_es_path = str(Path(__file__).resolve().parent / "data" / "es_index.txt")

# 知识图谱单例
_kg = None
_kg_failed = False
_graph_builder = None

# 术语倒排索引单例
_term_index = None
_term_index_path = str(Path(__file__).resolve().parent / "data" / "term_index.pkl")

# MySQL 单例
_db_available = False
_db_session = None

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _get_embedder():
    global _embedder
    if _embedder is None:
        from rag_finance_system.src.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        from rag_finance_system.src.vector_store import VectorStore
        _vector_store = VectorStore()
    return _vector_store


def _get_reranker():
    global _reranker, _reranker_failed
    if _reranker is None and not _reranker_failed:
        try:
            from rag_finance_system.src.embedder import Reranker
            _reranker = Reranker()
        except Exception as e:
            _reranker_failed = True
            logger.warning(f"Reranker 加载失败: {e}")
    return _reranker


def _get_bm25():
    global _bm25_index
    if _bm25_index is None:
        from rag_finance_system.src.bm25_index import BM25Index

        _bm25_index = BM25Index.load(_bm25_path)
        if _bm25_index is None:
            _bm25_index = BM25Index()
            logger.info("BM25 索引文件不存在，将创建空索引（纯向量模式）")
    return _bm25_index


def _get_es():
    global _es_index, _es_failed
    if _es_index is None and not _es_failed:
        try:
            from rag_finance_system.src.es_index import ESIndex

            _es_index = ESIndex()
            if not _es_index._connected:
                _es_failed = True
                _es_index = None
                logger.info("ES 不可用，将回退 BM25")
            else:
                logger.info(f"ES 全文索引就绪: {_es_index.doc_count} 篇")
        except Exception as e:
            _es_failed = True
            logger.warning(f"ES 初始化失败，将回退 BM25: {e}")
    return _es_index


def _get_term_index():
    global _term_index
    if _term_index is None:
        from rag_finance_system.src.term_index import TermIndex

        _term_index = TermIndex.load(_term_index_path)
        if _term_index is None:
            _term_index = TermIndex(dictionary=_get_dictionary())
            logger.info("术语倒排索引文件不存在，将创建空索引")
        else:
            _term_index._dictionary = _get_dictionary()
            logger.info(f"术语倒排索引已加载: {_term_index.doc_count} 条记录")
    return _term_index


def _get_processor():
    global _processor
    if _processor is None:
        from rag_finance_system.src.document_processor import DocumentProcessor
        _processor = DocumentProcessor()
    return _processor


def _get_dictionary():
    global _dictionary, _dictionary_failed
    if _dictionary is None and not _dictionary_failed:
        try:
            from rag_finance_system.src.dictionary import FinanceDictionary
            _dictionary = FinanceDictionary(dict_path=_dict_path)
            logger.info(f"金融词典已加载: {_dictionary.stats()}")
        except Exception as e:
            _dictionary_failed = True
            logger.warning(f"金融词典加载失败（系统将使用文件名索引回退）: {e}")
    return _dictionary


def _get_graph():
    global _kg, _kg_failed
    if _kg is None and not _kg_failed:
        try:
            from rag_finance_system.src.knowledge_graph import KnowledgeGraph
            _kg = KnowledgeGraph()
            if _kg._connected:
                logger.info(f"知识图谱已就绪: {_kg.stats()}")
            else:
                _kg_failed = True
                _kg = None
        except Exception as e:
            _kg_failed = True
            logger.warning(f"知识图谱初始化失败（将跳过图谱功能）: {e}")
    return _kg


def _get_graph_builder():
    global _graph_builder
    if _graph_builder is None:
        kg = _get_graph()
        if kg and kg._connected:
            from rag_finance_system.src.graph_builder import GraphBuilder
            _graph_builder = GraphBuilder(kg=kg, dictionary=_get_dictionary())
            _graph_builder.sync_dictionary_to_graph()
    return _graph_builder


def _get_db():
    """获取 MySQL session，失败时返回 None 以便降级"""
    global _db_available, _db_session
    if not _db_available:
        return None
    if _db_session is None:
        try:
            from sqlalchemy.orm import Session
            from rag_finance_system.src.database import SessionLocal
            _db_session = SessionLocal()
        except Exception as e:
            _db_available = False
            logger.warning(f"MySQL 会话创建失败: {e}")
            return None
    return _db_session


def _get_chat_store():
    db = _get_db()
    if db is None:
        return None
    from rag_finance_system.src.chat_store import ChatStore
    return ChatStore(db)


def _get_llm():
    global _llm
    if _llm is None:
        try:
            from rag_finance_system.src.llm import get_llm
            _llm = get_llm(prefer_local=False)
            logger.info(f"LLM 已就绪: {type(_llm).__name__}")
        except Exception as e:
            logger.warning(f"LLM 初始化失败: {e}")
            _llm = None
    return _llm


def _get_rag(use_api: bool = False):
    global _rag
    if _rag is None:
        from rag_finance_system.src.retriever import Retriever
        from rag_finance_system.src.llm import get_llm
        from rag_finance_system.src.rag_chain import RAGChain
        from rag_finance_system.src.rewriter import QueryRewriter

        embedder = _get_embedder()
        vs = _get_vector_store()
        reranker = _get_reranker()
        dictionary = _get_dictionary()

        rewriter = None
        try:
            lora_path = str(Path(__file__).resolve().parent.parent / "checkpoints" / "rewriter_lora" / "final")
            if Path(lora_path).exists():
                rewriter = QueryRewriter(model_path="Qwen/Qwen2.5-0.5B-Instruct", lora_path=lora_path)
            else:
                rewriter = QueryRewriter()
        except Exception as e:
            logger.warning(f"查询重写模型加载失败: {e}")

        llm = get_llm(prefer_local=not use_api)
        bm25 = _get_bm25()
        retriever = Retriever(
            embedder=embedder, vector_store=vs, reranker=reranker,
            bm25_index=bm25, es_index=_get_es(), term_index=_get_term_index(),
        )
        kg = _get_graph()
        _rag = RAGChain(retriever=retriever, llm=llm, rewriter=rewriter, dictionary=dictionary, knowledge_graph=kg)
    return _rag


def _check_milvus() -> bool:
    try:
        vs = _get_vector_store()
        vs.client.has_collection(vs.collection_name)
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_available
    logger.info("预加载 Embedder + VectorStore + BM25 + ES + 知识图谱 + 术语索引...")
    try:
        _get_embedder()
        _get_vector_store()
        _get_bm25()
        _get_es()
        _get_graph_builder()
        _get_term_index()
        logger.info("Embedder / VectorStore / BM25 / ES / 知识图谱 / 术语索引 就绪")
    except Exception as e:
        logger.error(f"启动预加载失败: {e}")

    if _check_milvus():
        logger.info("Milvus 连接正常")
    else:
        logger.warning("Milvus 未连接 — 上传/检索接口将返回 503")

    # 初始化 MySQL
    try:
        from rag_finance_system.src.database import init_db
        import rag_finance_system.src.models  # noqa: F401 — 确保表注册到 Base.metadata
        init_db()
        _db_available = True
        logger.info("MySQL 连接正常，对话历史/收藏功能已启用")
    except Exception as e:
        _db_available = False
        logger.warning(f"MySQL 不可用，对话历史/收藏功能将降级: {e}")

    yield


app = FastAPI(title="金融法规 RAG API", version="1.0.0", lifespan=lifespan)


# ── 1. 文档上传 ──

@app.post("/api/documents/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile = File(...),
    doc_type: str = Form("law"),
):
    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext}，仅支持 PDF/TXT")

    save_dir = Path("data/raw") / doc_type
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / file.filename

    try:
        content = file.file.read()
        save_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(500, f"文件保存失败: {e}")

    return UploadResponse(
        filename=file.filename,
        file_path=str(save_path),
        doc_type=doc_type,
        size_bytes=len(content),
    )


# ── 2. 建索引 ──

@app.post("/api/documents/index", response_model=IndexResponse)
def build_index(body: IndexRequest):
    if not _check_milvus():
        raise HTTPException(503, "Milvus 服务不可用")

    file_path = body.file_path
    if not os.path.exists(file_path):
        raise HTTPException(404, f"文件不存在: {file_path}")

    try:
        processor = _get_processor()
        chunks = processor.process_file(file_path, doc_type=body.doc_type)
    except Exception as e:
        raise HTTPException(500, f"文档解析失败: {e}")

    if not chunks:
        raise HTTPException(500, "解析结果为空")

    try:
        embedder = _get_embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_documents(texts, batch_size=16)
    except Exception as e:
        raise HTTPException(500, f"Embedding 失败: {e}")

    try:
        vs = _get_vector_store()
        vs.insert(chunks, embeddings)
    except Exception as e:
        raise HTTPException(500, f"Milvus 写入失败: {e}")

    try:
        bm25 = _get_bm25()
        bm25.index(chunks)
        bm25.save(_bm25_path)
    except Exception as e:
        logger.warning(f"BM25 索引更新失败（不影响向量检索）: {e}")

    try:
        es_idx = _get_es()
        if es_idx is not None:
            es_idx.index(chunks)
            es_idx.save(_es_path)
            logger.info(f"ES 索引已同步: {es_idx.doc_count} 篇")
    except Exception as e:
        logger.warning(f"ES 索引同步失败（不影响向量+BM25 检索）: {e}")

    try:
        term_idx = _get_term_index()
        if term_idx._dictionary is None:
            term_idx._dictionary = _get_dictionary()
        term_idx.index(chunks)
        term_idx.save(_term_index_path)
        logger.info(f"术语倒排索引已同步: {term_idx.doc_count} 条记录")
    except Exception as e:
        logger.warning(f"术语倒排索引同步失败（不影响向量+BM25 检索）: {e}")

    try:
        builder = _get_graph_builder()
        if builder is not None:
            graph_stats = builder.build_from_chunks(chunks)
            logger.info(f"知识图谱已同步: {graph_stats}")
    except Exception as e:
        logger.warning(f"知识图谱同步失败（不影响检索）: {e}")

    return IndexResponse(
        chunk_count=len(chunks),
        doc_type=body.doc_type,
        file_path=file_path,
    )


# ── 3. 检索 (无 LLM) ──

@app.post("/api/search", response_model=SearchResponse)
def search(body: SearchRequest):
    if not _check_milvus():
        raise HTTPException(503, "Milvus 服务不可用")

    from rag_finance_system.src.retriever import Retriever

    try:
        retriever = Retriever(
            embedder=_get_embedder(),
            vector_store=_get_vector_store(),
            reranker=_get_reranker() if body.use_reranker else None,
            bm25_index=_get_bm25(),
            es_index=_get_es(),
            term_index=_get_term_index(),
            top_k=body.top_k,
        )
        chunks = retriever.retrieve(
            query=body.query,
            top_k=body.top_k,
            use_reranker=body.use_reranker,
            doc_type_filter=body.doc_type_filter,
            law_name_filter=body.law_name_filter,
            authority_filter=body.authority_filter,
        )
    except Exception as e:
        raise HTTPException(500, f"检索失败: {e}")

    results = [
        SearchResultItem(
            text=c.get("text", ""),
            source=c.get("source", ""),
            article_num=c.get("article_num", ""),
            score=round(c.get("reranker_score", c.get("score", 0.0)), 4),
            law_name=c.get("law_name", ""),
            doc_type=c.get("doc_type", "law"),
        )
        for c in chunks
    ]

    return SearchResponse(query=body.query, results=results)


# ── 4. 问答 ──

@app.post("/api/qa", response_model=QAResponse)
def qa(body: QARequest):
    if not _check_milvus():
        raise HTTPException(503, "Milvus 服务不可用")

    try:
        rag = _get_rag()
        result = rag.query(
            question=body.question,
            use_reranker=body.use_reranker,
            use_query_rewrite=body.use_query_rewrite,
            doc_type_filter=body.doc_type_filter,
            max_new_tokens=body.max_new_tokens,
        )
    except Exception as e:
        raise HTTPException(500, f"问答失败: {e}")

    sources = [
        SourceItem(
            source=s["source"],
            article_num=s["article_num"],
            text=s["text"],
            score=s["score"],
        )
        for s in result.get("sources", [])
    ]

    conf = result.get("confidence", {})
    confidence = ConfidenceScores(
        total=conf.get("total", 0.0),
        retrieval=conf.get("retrieval", 0.0),
        coverage=conf.get("coverage", 0.0),
    )

    return QAResponse(
        question=result["question"],
        answer=result["answer"],
        rewritten_query=result.get("rewritten_query"),
        sources=sources,
        confidence=confidence,
    )


# ── 5. 对话历史 ──

@app.get("/api/conversations", response_model=list[ConversationOut])
def list_conversations():
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    convs = store.list_conversations()
    return [
        ConversationOut(
            id=c.id, title=c.title,
            created_at=c.created_at, updated_at=c.updated_at,
            message_count=store.get_message_count(c.id),
        )
        for c in convs
    ]


@app.post("/api/conversations", response_model=ConversationOut)
def create_conversation(body: ConversationCreate):
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    conv = store.create_conversation(title=body.title)
    return ConversationOut(
        id=conv.id, title=conv.title,
        created_at=conv.created_at, updated_at=conv.updated_at,
        message_count=0,
    )


@app.get("/api/conversations/{conv_id}", response_model=ConversationDetail)
def get_conversation(conv_id: str):
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "对话不存在")
    messages = store.get_messages(conv_id)
    return ConversationDetail(
        id=conv.id, title=conv.title,
        created_at=conv.created_at, updated_at=conv.updated_at,
        messages=[
            MessageOut(
                id=m.id, role=m.role, content=m.content,
                question=m.question, rewritten_query=m.rewritten_query,
                sources=m.sources, confidence=m.confidence,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    if not store.delete_conversation(conv_id):
        raise HTTPException(404, "对话不存在")
    return {"ok": True}


# ── 6. 收藏 ──

@app.post("/api/favorites", response_model=FavoriteOut)
def add_favorite(body: FavoriteCreate):
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    fav = store.add_favorite(
        fav_type=body.fav_type,
        conversation_id=body.conversation_id,
        message_id=body.message_id,
        source_data=body.source_data,
        note=body.note,
    )
    sd = fav.source_data
    if isinstance(sd, str):
        try:
            sd = json.loads(sd)
        except (json.JSONDecodeError, TypeError):
            pass
    return FavoriteOut(
        id=fav.id, fav_type=fav.fav_type,
        conversation_id=fav.conversation_id,
        message_id=fav.message_id,
        source_data=sd,
        note=fav.note,
        created_at=fav.created_at,
    )


@app.get("/api/favorites", response_model=list[FavoriteOut])
def list_favorites(fav_type: str | None = None):
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    favs = store.list_favorites(fav_type=fav_type)
    result = []
    for f in favs:
        sd = f.source_data
        if isinstance(sd, str):
            try:
                sd = json.loads(sd)
            except (json.JSONDecodeError, TypeError):
                pass
        result.append(
            FavoriteOut(
                id=f.id, fav_type=f.fav_type,
                conversation_id=f.conversation_id,
                message_id=f.message_id,
                source_data=sd,
                note=f.note,
                created_at=f.created_at,
            )
        )
    return result


@app.delete("/api/favorites/{fav_id}")
def delete_favorite(fav_id: str):
    store = _get_chat_store()
    if store is None:
        raise HTTPException(503, "MySQL 不可用")
    if not store.delete_favorite(fav_id):
        raise HTTPException(404, "收藏不存在")
    return {"ok": True}


# ── 7. 流式问答（支持对话历史） ──

@app.post("/api/qa/stream")
def qa_stream(body: QAStreamRequest):
    """流式 SSE 问答：检索后逐 token 推送，支持自动保存对话历史。"""
    if not _check_milvus():
        raise HTTPException(503, "Milvus 服务不可用")

    def _generate():
        rag = _get_rag()
        store = _get_chat_store()
        conv_id = body.conversation_id
        answer_text = ""
        meta_data = {}

        try:
            if store and not conv_id:
                conv = store.create_conversation(title=body.question[:50])
                conv_id = conv.id

            if store and conv_id:
                store.add_message(conv_id, role="user", content=body.question)

            for line in rag.query_stream(
                question=body.question,
                use_reranker=body.use_reranker,
                use_query_rewrite=body.use_query_rewrite,
                doc_type_filter=body.doc_type_filter,
                max_new_tokens=body.max_new_tokens,
            ):
                yield f"data: {line}\n\n"
                event = json.loads(line)
                if event.get("type") == "meta":
                    meta_data = event
                elif event.get("type") == "token":
                    answer_text += event.get("text", "")
                elif event.get("type") == "done":
                    meta_data["confidence"] = event.get("confidence", {})

            if store and conv_id:
                sources = meta_data.get("sources", [])
                if isinstance(sources, str):
                    sources = json.loads(sources)
                confidence = meta_data.get("confidence", {})
                if isinstance(confidence, str):
                    confidence = json.loads(confidence)
                store.add_message(
                    conv_id, role="assistant", content=answer_text,
                    question=body.question,
                    rewritten_query=meta_data.get("rewritten_query"),
                    sources=sources, confidence=confidence,
                )

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式问答失败: {e}")
            yield f"data: {{\"type\": \"error\", \"message\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 8. 分类管理 ──

@app.get("/api/categories", response_model=CategoryOut)
def get_categories():
    dict_obj = _get_dictionary()
    if dict_obj is None:
        return CategoryOut(categories={})
    raw = dict_obj._raw
    cats: dict[str, list[str]] = {}
    for section_key in ("terms", "law_names", "authorities"):
        for name, info in raw.get(section_key, {}).items():
            c = info.get("category", "")
            if c:
                label = "term" if section_key == "terms" else "law" if section_key == "law_names" else "authority"
                cats.setdefault(label, [])
                if c not in cats[label]:
                    cats[label].append(c)
    for k in cats:
        cats[k].sort()
    return CategoryOut(categories=cats)


@app.put("/api/categories/rename")
def rename_category(body: CategoryRename):
    dict_obj = _get_dictionary()
    if dict_obj is None:
        raise HTTPException(503, "词典不可用")
    raw = dict_obj._raw
    affected = {"term": 0, "law": 0, "authority": 0}
    for section_key, label in [("terms", "term"), ("law_names", "law"), ("authorities", "authority")]:
        for name, info in raw.get(section_key, {}).items():
            if info.get("category") == body.old_name:
                info["category"] = body.new_name
                affected[label] += 1
    _save_dictionary(raw)
    return {"ok": True, "affected": affected}


@app.delete("/api/categories/{name}")
def delete_category(name: str):
    dict_obj = _get_dictionary()
    if dict_obj is None:
        raise HTTPException(503, "词典不可用")
    raw = dict_obj._raw
    affected = {"term": 0, "law": 0, "authority": 0}
    for section_key, label in [("terms", "term"), ("law_names", "law"), ("authorities", "authority")]:
        for n, info in raw.get(section_key, {}).items():
            if info.get("category") == name:
                info["category"] = ""
                affected[label] += 1
    _save_dictionary(raw)
    return {"ok": True, "affected": affected}


def _save_dictionary(raw: dict):
    global _dictionary, _dictionary_failed
    try:
        with open(_dict_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        _dictionary = None
        _dictionary_failed = False
        logger.info("金融词典已保存并重新加载")
    except Exception as e:
        logger.error(f"保存词典失败: {e}")


# ── 9. 词典条目管理 ──

@app.get("/api/dictionary/{item_type}", response_model=DictionaryItemsResponse)
def get_dictionary_items(item_type: str):
    dict_obj = _get_dictionary()
    if dict_obj is None:
        raise HTTPException(503, "词典不可用")
    try:
        raw = dict_obj._raw
        section_map = {"term": "terms", "law": "law_names", "authority": "authorities"}
        section_key = section_map.get(item_type)
        if not section_key:
            raise HTTPException(400, f"无效的 item_type: {item_type}")
        items = []
        for name, info in raw.get(section_key, {}).items():
            items.append(DictionaryItemOut(
                name=name,
                definition=info.get("definition", ""),
                category=info.get("category", ""),
                aliases=info.get("aliases", []),
            ))
        return DictionaryItemsResponse(items=items)
    except Exception as e:
        logger.error(f"获取词典条目失败: {e}")
        raise HTTPException(500, f"获取词典条目失败: {e}")


@app.put("/api/dictionary/{item_name}/category")
def set_item_category(item_name: str, body: DictionaryCategoryUpdate):
    dict_obj = _get_dictionary()
    if dict_obj is None:
        raise HTTPException(503, "词典不可用")
    raw = dict_obj._raw
    section_map = {"term": "terms", "law": "law_names", "authority": "authorities"}
    section_key = section_map.get(body.item_type)
    if not section_key:
        raise HTTPException(400, f"无效的 item_type: {body.item_type}")
    entry = raw.get(section_key, {}).get(item_name)
    if entry is None:
        raise HTTPException(404, f"条目 '{item_name}' 不存在")
    entry["category"] = body.category
    _save_dictionary(raw)
    return {"ok": True}


# ── 10. 法规查询 ──

@app.get("/api/laws", response_model=LawNamesResponse)
def get_law_names():
    vs = _get_vector_store()
    if vs is None:
        return LawNamesResponse(law_names=[])
    try:
        stats = vs.get_collection_stats()
        law_names = stats.get("distinct_law_names", [])
        if not law_names:
            results = vs.search(query_text="", top_k=10000, output_fields=["law_name"])
            law_names = sorted(set(r.get("law_name", "") for r in results if r.get("law_name")))
        return LawNamesResponse(law_names=law_names)
    except Exception as e:
        logger.warning(f"获取法规列表失败: {e}")
        return LawNamesResponse(law_names=[])


# ── 11. 条文关联查询 ──

_CN_NUM_MAP = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
    "十": 10, "百": 100, "千": 1000,
}


def _parse_cn_number(text: str) -> str:
    text = text.strip()
    if text.isdigit():
        return text
    result = 0
    current = 0
    for ch in text:
        val = _CN_NUM_MAP.get(ch)
        if val is None:
            continue
        if val >= 10:
            current = (current or 1) * val
            result += current
            current = 0
        else:
            current = val
    result += current
    return str(result)


def _normalize_article_num(input_num: str) -> list[str]:
    raw = input_num.strip()
    if not raw:
        return []
    candidates: list[str] = [raw]
    bare = raw
    m = re.match(r"第(.+)条", raw)
    if m:
        bare = m.group(1)
    if bare.isdigit():
        candidates.append(bare)
        candidates.append(f"第{bare}条")
    elif bare != raw:
        candidates.append(bare)
        candidates.append(f"第{bare}条")
        parsed = _parse_cn_number(bare)
        if parsed != bare:
            candidates.append(parsed)
            candidates.append(f"第{parsed}条")
    else:
        parsed = _parse_cn_number(bare)
        if parsed != bare:
            candidates.append(parsed)
            candidates.append(f"第{parsed}条")
    return list(set(candidates))


_REF_PATTERN = re.compile(r"《([^》]{2,60})》\s*第\s*([零一二三四五六七八九十百千\d]+)\s*条")


def _query_article_fallback(law_name: str, article_num: str) -> dict:
    """When Neo4j is unavailable, query Milvus directly for article info."""
    try:
        vs = _get_vector_store()
        if vs is None or not vs.client.has_collection(vs.collection_name):
            return {}
    except Exception:
        return {}

    article_candidates = _normalize_article_num(article_num)

    # 1. Find target article
    target = None
    for art in article_candidates:
        escaped_law = law_name.replace("\\", "\\\\").replace('"', '\\"')
        escaped_art = art.replace("\\", "\\\\").replace('"', '\\"')
        expr = f'law_name == "{escaped_law}" and article_num == "{escaped_art}"'
        try:
            results = vs.client.query(
                collection_name=vs.collection_name,
                filter=expr,
                output_fields=["law_name", "article_num", "text", "source", "chunk_id", "doc_type"],
                limit=1,
            )
        except Exception:
            results = []
        if results:
            hit = results[0]
            target = {
                "law_name": hit.get("law_name", law_name),
                "article_num": hit.get("article_num", article_num),
                "text": hit.get("text", ""),
                "source": hit.get("source", ""),
            }
            break

    if not target:
        return {}

    result: dict = {"target": target}

    # 2. Outgoing refs — extract 《XXX》第X条 from target text
    outgoing_refs = []
    for match in _REF_PATTERN.finditer(target["text"]):
        ref_law = match.group(1).strip()
        ref_article_raw = match.group(2).strip()
        ref_article = _parse_cn_number(ref_article_raw)
        outgoing_refs.append({
            "law_name": ref_law,
            "article_num": f"第{ref_article}条",
            "text": "",
            "source": "",
        })
    result["outgoing_refs"] = outgoing_refs

    # 3. Incoming refs — search other chunks mentioning this article number
    incoming_refs = []
    seen_chunks: set[str] = set()
    for art in article_candidates:
        escaped_law_name = law_name.replace("\\", "\\\\").replace('"', '\\"')
        escaped_art = art.replace("\\", "\\\\").replace('"', '\\"')
        try:
            ref_results = vs.client.query(
                collection_name=vs.collection_name,
                filter='law_name != "' + escaped_law_name + '" and article_num == "' + escaped_art + '"',
                output_fields=["law_name", "article_num", "text", "source", "chunk_id"],
                limit=10,
            )
        except Exception:
            ref_results = []
        for hit in ref_results:
            cid = hit.get("chunk_id", "")
            if cid and cid not in seen_chunks:
                seen_chunks.add(cid)
                incoming_refs.append({
                    "law_name": hit.get("law_name", ""),
                    "article_num": hit.get("article_num", ""),
                    "text": hit.get("text", ""),
                    "source": hit.get("source", ""),
                })
    result["incoming_refs"] = incoming_refs

    # 4. Parent document
    result["parent_document"] = {
        "name": target.get("law_name", law_name),
        "doc_type": "law",
        "source": target.get("source", ""),
    }

    # 5. Related documents & articles — cannot infer without graph edges
    result["related_documents"] = []
    result["related_articles"] = []

    return result


@app.post("/api/articles/relations", response_model=ArticleRelationResponse)
def query_article_relations(body: ArticleRelationRequest):
    kg = _get_graph()
    if kg is not None and kg._connected:
        try:
            result = kg.get_article_relations(body.law_name, body.article_num)
            if result:
                return ArticleRelationResponse(**result)
        except Exception as e:
            logger.warning(f"Neo4j 条文关联查询失败，降级到 Milvus 查询: {e}")

    logger.info(f"使用 Milvus 降级查询: {body.law_name} 第{body.article_num}条")
    result = _query_article_fallback(body.law_name, body.article_num)
    return ArticleRelationResponse(**result)


# ── 12. 流程图生成 ──

@app.post("/api/flowchart/image", response_model=FlowchartResponse)
def flowchart_from_image(body: FlowchartImageRequest):
    try:
        import base64 as b64
        img_bytes = b64.b64decode(body.image_base64)
        ocr_text = ""
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name
            result = ocr.ocr(tmp_path, cls=True)
            os.unlink(tmp_path)
            if result and result[0]:
                ocr_text = "\n".join([line[1][0] for line in result[0] if line[1]])
        except Exception:
            pass
        if not ocr_text:
            return FlowchartResponse(success=False, error="OCR 未能识别图片内容")
        llm = _get_llm()
        if llm is None:
            return FlowchartResponse(success=False, error="LLM 不可用")
        prompt = f"从以下文本中提取流程步骤，生成 Mermaid 流程图代码。只返回 Mermaid 代码，不要其他内容。\n\n{ocr_text}"
        mermaid = llm.generate(prompt)
        mermaid = mermaid.strip()
        if "```" in mermaid:
            mermaid = mermaid.split("```")[1]
            if mermaid.startswith("mermaid"):
                mermaid = mermaid[7:]
            mermaid = mermaid.strip()
        return FlowchartResponse(success=True, mermaid=mermaid, source="ocr+llm")
    except Exception as e:
        logger.error(f"图片流程图生成失败: {e}")
        return FlowchartResponse(success=False, error=str(e))


@app.post("/api/flowchart/text", response_model=FlowchartResponse)
def flowchart_from_text(body: FlowchartTextRequest):
    try:
        llm = _get_llm()
        if llm is None:
            return FlowchartResponse(success=False, error="LLM 不可用")
        prompt = f"从以下法规文本中提取流程步骤，生成 Mermaid 流程图代码。只返回 Mermaid 代码，不要其他内容。\n\n{body.text}"
        mermaid = llm.generate(prompt)
        mermaid = mermaid.strip()
        if "```" in mermaid:
            mermaid = mermaid.split("```")[1]
            if mermaid.startswith("mermaid"):
                mermaid = mermaid[7:]
            mermaid = mermaid.strip()
        return FlowchartResponse(success=True, mermaid=mermaid, source="text")
    except Exception as e:
        logger.error(f"文本流程图生成失败: {e}")
        return FlowchartResponse(success=False, error=str(e))
