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

## 功能

- **多类型文档解析**：支持法条 (law)、案例 (case)、其他参考资料 (other) 三种类型，PDF / TXT 格式
- **智能分段策略**：
  - 法条：按"第XX条"结构切分，条文内部递归切分，永不跨条
  - 案例：按裁判文书标准段落结构切分
  - 其他：三级优先级切分（第X条 → 中文序号 一、二、三… → 纯递归）
- **实体感知检索**：从用户问题中自动识别文件名（《》内法规简称）、法律名称（公司法→中华人民共和国公司法）、监管机构（上海→上海银保监局），多过滤器 OR 组合检索
- **元数据增强**：每个 chunk 自动提取法律名称 (law_name) 和发布机构 (authority)，支持精确过滤
- **查询重写**：优先使用 Qwen2.5-0.5B + LoRA 微调模型，失败回退主 LLM；支持侧边栏开关
- **四种检索模式**：全部 / 仅法条 / 仅案例 / 仅其他
- **答案溯源**：每条回答附带来源文件、条文编号和相关度评分（绿/橙/红三色标识）
- **可信度评分**：综合检索相关性（60%）与答案覆盖度（40%）
- **多 LLM 后端**：本地 Qwen2.5-7B-Int4 / DeepSeek API / 通义千问 API，自动降级
- **对话历史**：MySQL 持久化存储所有对话记录，支持查看、切换、删除
- **收藏功能**：收藏整个对话或单条溯源条文，支持查看和取消收藏
- **Docker 一键部署**：Docker Compose 编排 7 个服务（Milvus + MySQL + Neo4j + etcd + MinIO + API + 前端），GPU 支持
- **批量导入**：支持一键导入 testfiles 中的 148 份监管规范性文件

## 技术栈

| 组件 | 方案 |
|------|------|
| 前端 | Streamlit |
| 后端 | FastAPI (Uvicorn) |
| Embedding | bge-small-zh-v1.5 (512d) |
| Reranker | bge-reranker-v2-m3 (Cross-Encoder + Sigmoid) |
| 查询重写 | Qwen2.5-0.5B-Instruct + LoRA |
| 向量数据库 | Milvus (本地/自建服务) |
| 关系数据库 | MySQL 8.0 (对话历史/收藏) |
| 知识图谱 | Neo4j 5-Community (条文引用关系/法规关联) |
| LLM | Qwen2.5-7B-Instruct-GPTQ-Int4 / DeepSeek / 通义千问 |
| 文档解析 | pdfplumber (PDF) + 自研分段器 |
| 容器化 | Docker Compose + NVIDIA GPU |

## 项目结构

```
rag_finance_system/
├── README.md
├── LOCAL_DEVELOPMENT.md          # 本地开发功能详细文档
├── requirements.txt
├── .gitignore
├── .dockerignore
├── Dockerfile                       # Docker 镜像定义 (CUDA + GPU)
├── docker-compose.yml               # 一键部署 7 个服务
├── checkpoints/
│   └── rewriter_lora/               # 查询重写器 LoRA 微调权重
│       ├── checkpoint-136/
│       ├── checkpoint-340/
│       └── final/                   # 最终版权重
├── docker/
│   └── requirements-docker.txt      # Docker 精简依赖 (31 个包)
├── scripts/
│   └── docker-entrypoint.sh         # Docker 启动脚本
├── data/
│   ├── finance_dictionary.json      # 金融词典 (术语/法规名/机构名)
│   ├── bm25_index.pkl               # BM25 持久化索引
│   ├── questions.json               # 600 条测试问答对
│   ├── raw/                         # 上传文档存储
│   │   ├── law/                     # 法条原文
│   │   ├── case/                    # 案例原文
│   │   └── other/                   # 其他参考资料
│   └── testfiles/                   # 148 份地方监管规范性文件
│       ├── 上海监管局/
│       ├── 江苏监管局/
│       └── 浙江监管局/
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

```bash
# 模型路径
EMBEDDING_MODEL_PATH=./models/bge-small-zh-v1.5
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
LLM_MODEL_PATH=./models/Qwen2.5-7B-Int4

# API 密钥（本地模型不可用时自动切换）
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=xxx

# Milvus 连接配置（本地/自建服务）
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=finance_regulations
MILVUS_EMBED_DIM=512

# Neo4j 知识图谱配置（条文引用关系 / Docker 端口 17687→7687）
NEO4J_URI=bolt://localhost:17687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j
NEO4J_DATABASE=neo4j

# MySQL 连接配置（对话历史/收藏）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=rag_user
MYSQL_PASSWORD=rag123456
MYSQL_DATABASE=rag_finance

# 检索参数
RETRIEVER_TOP_K=10      # 向量检索召回数量
RERANKER_TOP_N=5        # Reranker 后保留数量
CHUNK_SIZE=512          # 分段字符数
CHUNK_OVERLAP=100       # 分段重叠字符数
```

## 使用

### 配置

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Milvus（需要 docker）
docker compose up -d milvus etcd minio

# 启动 MySQL（需要 docker）
docker compose up -d mysql

# 启动 Milvus + MySQL + Neo4j（需要 docker）
docker compose up -d milvus mysql neo4j

# 启动 FastAPI 后端（Windows 端口 8000 被保留时使用 9099）
uvicorn rag_finance_system.api_app:app --host 0.0.0.0 --port 8000

# 启动 Streamlit 前端
streamlit run rag_finance_system/app.py
```

### 导入文档

```bash
# 方式1：前端上传
# 启动后在侧边栏选择文件上传

# 方式2：命令行批量导入
python rag_finance_system/tools/import_testfiles.py
```

启动后访问：
- **Streamlit 前端**: http://localhost:8501
- **Neo4j Browser** (图谱可视化): http://localhost:7474
- **MinIO 控制台**: http://localhost:9001

### 前端操作流程

1. **上传文档**：侧边栏三个独立上传区分别对应法条、案例、其他资料
2. **建立索引**：点击对应"解析并建立索引"按钮
3. **选择模式**：全部 / 仅法条 / 仅案例 / 仅其他
4. **提问**：输入自然语言问题，系统自动进行实体检测和查询重写
5. **查看结果**：答案附带溯源条文（可展开）和可信度评分
6. **对话历史**：侧边栏显示历史对话列表，点击切换查看
7. **收藏功能**：对话中可收藏整个对话或单条溯源条文
8. **条文关联查询**（Tab 2）：输入法规名称和条文编号，查看引用关系网络（基于 Neo4j 知识图谱，Neo4j 不可用时降级为 Milvus 文本匹配）
9. **标签分类管理**（Tab 3）：管理金融词典中术语、法规和机构的分类标签
10. **侧边栏选项**：可切换 API 模式、开关 Reranker、开关查询重写

### 命令行工具

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

- ✅ FastAPI 后端 + Streamlit 前端
- ✅ MySQL 对话历史 + 收藏功能
- ✅ Docker Compose 一键部署 (GPU + Qwen2.5-7B-Int4)
- ✅ Elasticsearch BM25 倒排索引 + 混合检索 (RRF 融合)
- ✅ Neo4j 知识图谱（条文引用关系网络 + 条文关联查询）
- OCR 增强管线（扫描件支持）
