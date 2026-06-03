# 金融制度知识问答系统 (RAG Finance System)

基于 RAG（检索增强生成）的金融法规智能问答系统，支持法条、案例、其他参考资料三类文档的智能检索、实体感知过滤、查询重写、语义问答、条文溯源与可信度评估。

## 架构概览

```
用户问题
  │
  ├── 实体检测 (文件名 / 法律名称 / 监管机构自动识别)
  ├── 查询重写 (Qwen2.5-0.5B + LoRA → 回退主 LLM)
  ├── 多过滤器向量检索 (bge-small-zh-v1.5 + ChromaDB)
  │     └── 法律名 / 机构 / 来源文件 OR 组合 + doc_type AND 过滤
  ├── Reranker 精排 (bge-reranker-v2-m3, Sigmoid 归一化)
  ├── Prompt 组装 (System + Context + Question)
  ├── LLM 生成 (Qwen2.5-7B / DeepSeek / 通义千问, 自动降级)
  └── 可信度评分 + 溯源展示 + 改写查询回显
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
- **批量导入**：支持一键导入 testfiles 中的 148 份监管规范性文件

## 技术栈

| 组件 | 方案 |
|------|------|
| 前端 | Streamlit |
| Embedding | bge-small-zh-v1.5 (512d) |
| Reranker | bge-reranker-v2-m3 (Cross-Encoder + Sigmoid) |
| 查询重写 | Qwen2.5-0.5B-Instruct + LoRA |
| 向量数据库 | ChromaDB (本地持久化) |
| LLM | Qwen2.5-7B-Instruct-GPTQ-Int4 / DeepSeek / 通义千问 |
| 文档解析 | pdfplumber (PDF) + 自研分段器 |

## 项目结构

```
rag_finance_system/
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/
│   └── rewriter_lora/            # 查询重写器 LoRA 微调权重
│       ├── checkpoint-136/
│       ├── checkpoint-340/
│       └── final/                # 最终版权重
├── data/
│   ├── questions.json            # 600 条测试问答对 (question + query)
│   ├── raw/                      # 上传文档存储
│   │   ├── law/                  # 法条原文
│   │   ├── case/                 # 案例原文
│   │   └── other/                # 其他参考资料
│   └── testfiles/                # 148 份地方监管规范性文件
│       ├── 上海监管局/
│       ├── 江苏监管局/
│       └── 浙江监管局/
├── db/chroma/                    # ChromaDB 向量存储
├── rag_finance_system/
│   ├── .env                      # 环境配置 (不纳入版本控制)
│   ├── app.py                    # Streamlit 前端
│   ├── src/
│   │   ├── __init__.py
│   │   ├── document_processor.py # 文档解析 + 三轨智能分段
│   │   ├── embedder.py           # Embedding + Reranker (Sigmoid)
│   │   ├── vector_store.py       # ChromaDB 向量存储 (多维度过滤)
│   │   ├── retriever.py          # 检索器 (多过滤器 OR + 精排 + 可信度)
│   │   ├── rag_chain.py          # RAG 主链路 (实体检测 → 重写 → 检索 → 生成 → 溯源)
│   │   ├── rewriter.py           # 查询重写器 (小模型 + LoRA)
│   │   ├── llm.py                # LLM 推理 (本地/API 三路 + 自动降级)
│   │   ├── change.py             # docx → txt 转换工具
│   │   └── txt_files/            # 83 部中国金融法律原文
│   └── tools/
│       ├── convert_testfiles.py  # doc/docx → txt 批量转换 (LibreOffice)
│       ├── generate_questions.py # 调用 API 从文档自动生成测试问题
│       ├── generate_rewrite_data.py # 生成查询重写训练数据
│       ├── rewrite_questions_for_rag.py # 批量重写问题为检索查询
│       ├── train_rewriter.py     # LoRA 微调查询重写小模型
│       └── import_testfiles.py   # 批量导入 testfiles 到向量库
├── models/                       # 本地模型 (不纳入版本控制)
│   ├── bge-small-zh-v1.5/
│   └── bge-reranker-v2-m3/
└── download_model.py             # 模型下载/验证脚本
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/intnerd/rag_finance_system.git
cd rag_finance_system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或 venv\Scripts\activate  (Windows)

# 安装依赖
pip install -r requirements.txt

# 下载模型（如未包含在本地）
# Embedding:  BAAI/bge-small-zh-v1.5
# Reranker:   BAAI/bge-reranker-v2-m3
# LLM:        Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
# Rewriter:   Qwen/Qwen2.5-0.5B-Instruct (自动从 HuggingFace 下载)
```

## 配置

复制并编辑 `.env` 文件（仓库已含模板，API Key 需自行填写）：

```bash
# 模型路径
EMBEDDING_MODEL_PATH=./models/bge-small-zh-v1.5
RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3
LLM_MODEL_PATH=./models/Qwen2.5-7B-Int4

# API 密钥（本地模型不可用时自动切换）
DEEPSEEK_API_KEY=sk-xxx
DASHSCOPE_API_KEY=xxx

# 检索参数
RETRIEVER_TOP_K=10      # 向量检索召回数量
RERANKER_TOP_N=5        # Reranker 后保留数量
CHUNK_SIZE=512          # 分段字符数
CHUNK_OVERLAP=100       # 分段重叠字符数
```

## 使用

### 启动前端

```bash
streamlit run rag_finance_system/app.py
```

### 前端操作流程

1. **上传文档**：侧边栏三个独立上传区分别对应法条、案例、其他资料
2. **建立索引**：点击对应"解析并建立索引"按钮
3. **选择模式**：全部 / 仅法条 / 仅案例 / 仅其他
4. **提问**：输入自然语言问题，系统自动进行实体检测和查询重写
5. **查看结果**：答案附带溯源条文（可展开）和可信度评分
6. **侧边栏选项**：可切换 API 模式、开关 Reranker、开关查询重写

### 命令行工具

```bash
# 将 testfiles 中的 .doc/.docx 文件转换为 .txt
python rag_finance_system/tools/convert_testfiles.py

# 批量导入 testfiles 到向量库
python rag_finance_system/tools/import_testfiles.py

# 从文档自动生成测试问题
python rag_finance_system/tools/generate_questions.py

# 批量重写问题为检索查询
python rag_finance_system/tools/rewrite_questions_for_rag.py

# 生成查询重写训练数据
python rag_finance_system/tools/generate_rewrite_data.py --mode api

# LoRA 微调查询重写模型
python rag_finance_system/tools/train_rewriter.py
```

## 文档格式要求

### 法条文档 (doc_type=law)

无特殊格式要求，系统会自动识别"第XX条"结构并按条切分。支持《中华人民共和国XX法》格式的法律文件。

### 案例文档 (doc_type=case)

需遵循中国裁判文书标准格式（最高人民法院法〔2016〕221 号规范），包含以下结构段落：

- `当事人信息：` / `原告诉称：` / `被告辩称：`
- `诉讼请求：`
- `事实与理由：`
- `经审理查明：`
- `本院认为：`
- `判决如下：` / `裁定如下：`

来自中国裁判文书网 (wenshu.court.gov.cn) 或北大法宝等数据库的文书天然符合此格式。

### 其他参考资料 (doc_type=other)

适用于学术文献、研究报告、监管通知、指导意见、政策解读等。系统自动选择最优切分策略：
1. 优先按"第X条"切分（适用于管理办法/实施细则类）
2. 否则按中文序号"一、二、三…"切分（适用于通知/指导意见类）
3. 兜底按固定字符数递归切分（适用于公告/模板类）

## 工作流程

```
文档上传 → PDF/TXT 解析 → 智能分段（法条/案例/其他三轨）
  → 元数据提取（法律名、发布机构） → Embedding → ChromaDB 存储

用户提问 → 实体检测（文件名/法律名/机构）
  → 查询重写（小模型 + LoRA → 回退主 LLM）
  → 多过滤器向量检索 (Top-10, 维度 OR + 类型 AND)
  → Reranker 精排 (Top-5, Sigmoid 归一化)
  → Prompt 组装 (System + Context + Question)
  → LLM 生成（本地 → DeepSeek → 通义千问 自动降级）
  → 答案 + 溯源 + 可信度 + 改写查询回显
```

## 创新点

1. **实体感知检索**：从自然语言问题中自动识别法律名称、监管机构、文件名称，无需用户手动筛选
2. **三轨文档处理**：法条、案例、其他参考资料各自采用最优切分策略，元数据自动提取
3. **查询重写 + 微调**：专用小模型 + LoRA 微调，高效改写口语化问题为检索友好查询
4. **多维度溯源**：每条答案附带来源文件、条文编号、相关度评分（绿/橙/红三色可视标识）
5. **可信度评分**：检索相关性 × 答案覆盖度双重评估，辅助用户判断是否需人工复核

## 路线图

参见 [rag金融知识技术路线.md](rag_finance_system/rag金融知识技术路线.md) 了解后续规划：

- Elasticsearch BM25 倒排索引 + 混合检索 (RRF 融合)
- Neo4j 知识图谱（条文引用关系网络）
- OCR 增强管线（扫描件支持）
- MySQL 元数据管理 + 时效性/废止检测
- FastAPI 后端 + Vue 3 前端
- Docker Compose 一键部署
