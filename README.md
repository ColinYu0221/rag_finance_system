# 金融制度知识问答系统 (RAG Finance System)

基于 RAG（检索增强生成）的金融法规智能问答系统，支持法条、案例、其他资料三类文档的智能检索、实体感知过滤、查询重写、流式问答、条文溯源与可信度评估。

## 架构总览

```
                        离线入库（一次性）
        PDF/TXT → 三轨分段 → 元数据提取 → Embedding → Milvus
                           ↓
                    2500+ 结构化 Chunk
                           ↓
        ┌──────────────────┴──────────────────┐
        │          在线问答（每次提问）           │
        │  用户问题                              │
        │    → 实体检测（文件/法律/机构 自动识别）   │
        │    → 查询重写（0.5B 小模型 + LoRA 微调）  │
        │    → 三路并行检索（向量 + BM25 + 术语索引）│
        │    → 加权 RRF 融合（BM25 权重 3×）       │
        │    → Reranker 精排（Cross-Encoder）     │
        │    → LLM 流式生成（本地 7B / API 降级）   │
        │    → 答案 + 溯源 + 可信度                 │
        └──────────────────────────────────────────┘
```

## 核心功能

| 模块 | 功能 | 技术要点 |
|------|------|---------|
| **三轨文档分段** | 按文档类型自适配切分策略 | law：强制按"第X条"；case：段落结构；other：第X条→一、二、三→递归 |
| **元数据自动提取** | 自动标注四维结构化标签 | law_name、authority、effective_date、status，全从文件名解析 |
| **实体检测** | 从口语问题中识别文件/法律/机构 | 金融词典(500+别名) + 文件扫描索引 + 公文规则，四层回退 |
| **查询重写** | 口语转检索语 | Qwen2.5-0.5B + LoRA 微调（600条数据），三级降级保障 |
| **三路并行检索** | 向量 + BM25 + 术语索引并发检索 | ThreadPoolExecutor，取最长耗时而非累加 |
| **加权 RRF 融合** | 多路召回融合 | BM25 权重 3×（金融术语精确匹配优先） |
| **流式 SSE 输出** | Token-by-token 生成 | TextIteratorStreamer + 后台线程 + `/api/qa/stream` |
| **检索过滤软回退** | 精确过滤0召回时自动降级 | source→查询增强，authority/status 过滤失败自动去掉 |
| **多版本自动管理** | 同名法规按日期判定有效/已修订 | 检索默认只看最新版本 |
| **文档 OCR** | 扫描件 PDF 文本提取 | docling 后端，2.0x 缩放提升中文识别率 |

## 技术栈

| 组件 | 方案 | 备注 |
|------|------|------|
| 前端 | Streamlit | 三类型独立上传、多文件选择、流式逐字渲染 |
| API 层 | FastAPI + Uvicorn | 6 个 REST 端点 + SSE 流式 |
| Embedding | bge-small-zh-v1.5 (512d) | BGE 指令前缀（查询编码加前缀） |
| Reranker | bge-reranker-v2-m3 | Cross-Encoder + Sigmoid + FP16 + Warmup |
| 向量数据库 | Milvus (pymilvus + milvus-lite) | AUTOINDEX / COSINE，嵌入式本地部署 |
| 关键词检索 | 内存 BM25 | jieba 中文分词，k1=1.5, b=0.75 |
| 全文检索 | Elasticsearch 8.x（可选） | IK 中文分词器，不可用时自动回退 BM25 |
| 术语索引 | 自研倒排索引 | 基于金融词典的精确术语匹配 |
| 查询重写 | Qwen2.5-0.5B-Instruct + LoRA | GPU/CPU 均可，<100ms |
| LLM | Qwen2.5-7B-Instruct-GPTQ-Int4 (5.3GB) | 本地 GPU 优先 → DeepSeek API → 通义千问 |
| 金融词典 | 76术语 + 55法律 + 29机构 + 500+别名 | JSON 格式，最长子串优先匹配 |
| 知识图谱 | Neo4j（可选） | 4节点3边类型，纯规则构建，不可用静默跳过 |
| 对话历史 | MySQL + SQLAlchemy（可选） | conversations / messages / favorites 三表 |
| OCR | docling（可选） | 自动激活（文本层<80字符或空白页>50%） |
| 文档解析 | pdfplumber + 自研分段器 | 支持 PDF / TXT |

## 项目结构

```
rag_finance_system/
├── README.md
├── LOCAL_DEVELOPMENT.md          # 本地开发功能详细文档
├── requirements.txt
├── docker-compose.yml            # Milvus + ES + Neo4j + MySQL
├── Dockerfile
├── checkpoints/rewriter_lora/    # 查询重写 LoRA 权重
├── data/
│   ├── finance_dictionary.json   # 金融词典
│   ├── dictionary_candidates.json
│   ├── questions.json             # 600条测试问答对
│   ├── raw/                      # 上传文档存储
│   └── testfiles/                # 148份地方规范性文件
├── db/milvus_finance.db/         # Milvus Lite 本地数据库
├── rag_finance_system/
│   ├── .env                      # 本地配置（不纳入版本控制）
│   ├── .env.example              # 配置模板
│   ├── api_app.py                # FastAPI 后端（6个端点+SSE）
│   ├── api_schemas.py            # Pydantic 数据模型
│   ├── app.py                    # Streamlit 前端
│   ├── test_retrieval_baseline.py
│   └── src/
│       ├── document_processor.py # PDF/TXT解析 + 三轨智能分段
│       ├── embedder.py           # Embedding + Reranker
│       ├── vector_store.py       # Milvus 向量存储
│       ├── bm25_index.py         # BM25 关键词索引
│       ├── es_index.py           # Elasticsearch 全文索引
│       ├── term_index.py         # 术语倒排索引
│       ├── retriever.py          # 三路并行检索 + 加权RRF + Reranker
│       ├── rag_chain.py          # RAG 主链路
│       ├── llm.py                # LLM 推理（本地+API三路降级）
│       ├── rewriter.py           # 查询重写小模型 + LoRA
│       ├── dictionary.py         # 金融词典查询引擎
│       ├── knowledge_graph.py    # Neo4j 知识图谱
│       ├── graph_builder.py      # 图谱纯规则构建器
│       ├── chat_store.py         # 对话历史业务逻辑
│       ├── database.py           # MySQL 连接管理
│       └── models.py             # SQLAlchemy ORM 模型
└── models/                       # 本地模型（建议放D盘）
    ├── bge-small-zh-v1.5/        # 92MB
    ├── bge-reranker-v2-m3/       # 2.2GB
    └── Qwen2.5-7B-Int4/          # 5.3GB
```

## 快速开始

### 环境准备

```bash
git clone https://github.com/intnerd/rag_finance_system.git
cd rag_finance_system
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

### 模型下载

将以下模型放入 `models/` 目录（或任意路径，在 `.env` 中配置）：

- Embedding: `BAAI/bge-small-zh-v1.5`
- Reranker: `BAAI/bge-reranker-v2-m3`
- LLM: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`
- Rewriter: `Qwen/Qwen2.5-0.5B-Instruct`（自动从 HuggingFace 下载）

### 配置

```bash
cp rag_finance_system/.env.example rag_finance_system/.env
# 编辑 .env 配置模型路径和 API 密钥
```

### 导入文档

```bash
# 方式1：前端上传
# 启动后在侧边栏选择文件上传

# 方式2：命令行批量导入
python rag_finance_system/tools/import_testfiles.py
```

### 启动

```bash
# 终端 1 — 后端
uvicorn rag_finance_system.api_app:app --host 0.0.0.0 --port 8000

# 终端 2 — 前端
streamlit run rag_finance_system/app.py

# 浏览器打开 http://localhost:8501
```

### Docker 部署（含 Milvus + ES + Neo4j + MySQL）

```bash
docker compose up -d
```

## API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/api/documents/upload` | 文件上传 |
| POST | `/api/documents/index` | 解析 + 入库 |
| POST | `/api/search` | 纯检索（无 LLM） |
| POST | `/api/qa` | 完整问答 |
| POST | `/api/qa/stream` | 流式问答（SSE） |
| GET | `/api/conversations` | 对话列表 |
| POST | `/api/conversations` | 创建对话 |
| GET | `/api/conversations/{id}` | 对话详情 |
| DELETE | `/api/conversations/{id}` | 删除对话 |
| POST | `/api/favorites` | 添加收藏 |
| GET | `/api/favorites` | 收藏列表 |

## 关键设计决策

**为什么 source_filter 改成了关键词注入？** 精确过滤匹配知识库中不存在的文件名时直接 0 召回。改为将检测到的文件名追加到查询文本中，让向量/BM25 模糊匹配能找到引用过该文件的其他文档。

**为什么 authority/law_name 过滤做了 0 召回自动回退？** 词典将 "中国保监会" 映射到 "国家金融监督管理总局"，但知识库中存储的是 "上海银保监局" 等地方机构名。过滤导致零结果时自动降级为无过滤检索。

**为什么 BM25 权重 3× 高于向量检索？** 金融法规中 "保险中介""分类监管""A类标准" 等精确术语匹配比语义向量更可靠。BM25 对专有名词的命中精度显著高于 512 维向量。

**为什么用 0.5B 而不是 7B 做查询重写？** 重写是表面语言变换（去口语噪声 + 补全实体名），不需要深度推理。0.5B 单次推理 <100ms，7B 则需要数秒。

**为什么用 milvus-lite 而不是 milvus-standalone？** 减少部署依赖。lite 版本嵌入 Python 进程，像 SQLite 一样零配置运行。适用于百万级以下向量的场景。

## License

MIT
