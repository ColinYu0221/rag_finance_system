# rag_finance_system — 项目上下文

## 项目概述

基于 RAG 的金融法规智能问答系统。Streamlit + FastAPI + Milvus + BM25，无外部框架依赖。

- Python: **3.12.3** (`py -3`)
- 入口: `rag_finance_system/app.py` (Streamlit)
- 测试: `rag_finance_system/test_pipeline.py`

## 核心调用链

```
app.py → api_app.py → load_components() → RAGChain.query()
  → 词典实体检测 → 查询扩展 → 查询重写 → Retriever.retrieve()
      → VectorStore.search() + BM25Index.search() → RRF融合 → Reranker精排
  → build_prompt() → LLM.generate() → 溯源 + 可信度评分
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `src/dictionary.py` | 金融词典：术语归一/别名召回/实体检测/查询扩展 |
| `src/rag_chain.py` | 主链路编排，RAGChain 类 |
| `src/retriever.py` | 向量+BM25双路召回 + RRF融合 + Reranker精排 |
| `src/bm25_index.py` | BM25关键词索引：jieba分词 + pickle持久化 |
| `src/vector_store.py` | Milvus适配层：insert/search/stats |
| `src/document_processor.py` | PDF/TXT解析 + 三轨智能分段 |
| `src/embedder.py` | bge-small-zh-v1.5 Embedding + bge-reranker-v2-m3 |
| `src/llm.py` | LLM工厂：本地Qwen → DeepSeek API → Qwen API |
| `src/rewriter.py` | 查询重写 (Qwen2.5-0.5B + LoRA) |
| `app.py` | Streamlit 前端 |
| `api_app.py` | FastAPI 后端 |

## Milvus 配置 (.env)

```
MILVUS_HOST=127.0.0.1, MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=finance_regulations
MILVUS_EMBED_DIM=512
```

## 已知问题

- `requirements.txt` 含未使用依赖: `faiss-cpu`, `langchain*`
- `download_model.py` 硬编码路径
- 无 `.env.example` 模板

## 环境

- OS: Windows 11
- pymilvus: 3.0.0
- 依赖已安装
