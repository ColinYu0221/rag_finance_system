# 本地开发功能详细文档

本文档记录了金融法规 RAG 问答系统所有本地独立开发的功能模块，按系统层级从下到上组织。

---

## 1. 三轨文档分段 (`document_processor.py`)

### 1.1 设计动机

金融法规系统需处理三类格式迥异的文档。统一按固定字符数切分会导致：

- 法律条文被从中间切断，一条法规跨两个 Chunk
- 监管通知的一、二、三…层级被打乱
- 检索时可能只命中半条条文，LLM 拿到不完整上下文

### 1.2 实现

在 `RecursiveCharacterTextSplitter` 类中，根据 `doc_type` 参数选择不同切分策略：

```
doc_type = "law" → 强制按"第X条"切割
doc_type = "case" → 标准段落递归切分
doc_type = "other" → 三级自适配
```

**law 策略** — `split_text()`：
```python
# 正则匹配所有条文块，按条切分
pattern = r"(第[零一二三四五六七八九十百千万\d]+条[^第]*)"
# 每条 ≤512 字符 → 直接输出
# 每条 >512 字符 → 条内按分隔符递归切分，永不跨条
```

**other 策略** — `split_text_for_other()`：
```python
# 第1级: _split_by_article() → 匹配"第X条"
# 第2级: _split_by_chinese_numbers() → 匹配"一、""二、""（一）""（二）"
# 第3级: _split_recursive() → 分隔符优先级 \n\n → \n → 。→ ；→ ，
```

**关键细节**：
- `_split_by_chinese_numbers()` 支持两种模式的正则匹配
- `_section_chunks()` 保证每个结构段落内部独立切分，不跨段落边界
- 分隔符优先级表：`["\n\n", "\n", "。", "；", "！", "？", "，", " "]`
- 参数：`chunk_size=512`, `chunk_overlap=100`, `keep_separator=True`

### 1.3 代码位置

| 方法 | 行号参考 | 功能 |
|------|---------|------|
| `RecursiveCharacterTextSplitter.split_text()` | ~50 | law 主切分入口 |
| `split_text_for_other()` | ~69 | other 三级自适配入口 |
| `_split_by_chinese_numbers()` | ~98 | 中文序号切分 |
| `_split_by_article()` | ~125 | 按条切分 |
| `_split_recursive()` | ~132 | 分隔符递归切分 |
| `_section_chunks()` | ~84 | 段落内独立处理 |

---

## 2. 元数据自动提取 (`document_processor.py`)

### 2.1 设计动机

向量检索只能做语义匹配。用户问 "上海怎么管保险中介的"，"上海" 这个地域信息丢失了就检索不到正确的文档。需要给每个 Chunk 打上结构化标签，查询时用这些标签做标量过滤。

### 2.2 四个自动提取字段

全部从文件名解析，零人工标注：

| 字段 | 提取方法 | 示例输入 | 示例输出 |
|------|---------|---------|---------|
| `law_name` | `re.sub(r"_\d{8}$", "", title)` 去掉日期后缀 | `中华人民共和国公司法_20231229.txt` | `中华人民共和国公司法` |
| `effective_date` | `re.search(r"_(\d{4})(\d{2})(\d{2})$")` 日期后缀转标准格式 | `_20231229` | `2023-12-29` |
| `authority` | `split("关于")[0]` 提取发文机构，去掉 "办公室"/"秘书处" 后缀 | `上海银保监局办公室关于印发《...》的通知.txt` | `上海银保监局` |
| `status` | 默认为 `"有效"`，批量导入后调用 `resolve_version_status()` 修正 | — | `"有效"` / `"已修订"` |

### 2.3 多版本自动管理

`resolve_version_status()` 静态方法：

```python
# 按 law_name 分组 → effective_date 降序排列
# 每组中最新日期 → status = "有效"，其余 → "已修订"
# 检索时默认 WHERE status = '有效'
```

同名法律多次修订入库后，用户默认只看到最新版本，旧版自动隐藏。可通过 `include_historical=True` 展开全部。

### 2.4 authority 提取的回退策略

当文件名中没有 "关于" 关键字时（无法直接提取机构名），从目录路径推断：

```python
# "金融监督管理局/上海监管局/规范性文件/..."
# 检测含"监管局"/"监管分局"的路径片段 → "上海监管局"
```

### 2.5 代码位置

| 方法 | 功能 |
|------|------|
| `DocumentProcessor.split_into_chunks()` | 调用元数据提取，写入每个 chunk |
| `_extract_law_name_and_date()` | 同时提取 law_name + effective_date |
| `_extract_date_from_title()` | 日期后缀 → YYYY-MM-DD |
| `_extract_authority()` | 两阶段机构名提取 |
| `resolve_version_status()` | 静态方法，多版本自动判定 |

---

## 3. 实体检测 (`rag_chain.py`)

### 3.1 设计动机

用户说 "上海的保险中介管理办法"，系统需要自动识别出：
- "上海" = 上海银保监局（监管机构）
- "保险中介管理办法" = 一份具体文件的简称

如果不做处理，向量检索会搜出所有包含 "上海" 和 "保险" 的文档（包括不相关的），召回精度极低。

### 3.2 四层检测架构

```
用户问题
  │
  ├── 第0层: 金融词典 (FinanceDictionary)
  │   ├─ 最长子串优先匹配算法
  │   ├─ 76 规范术语 + 357 别名
  │   ├─ 55 法律名称 + 167 别名
  │   └─ 29 监管机构 + 97 别名
  │
  ├── 第1层: 文件名检测 (_detect_source)
  │   ├─ 扫描 txt_files/ + testfiles/ 自动构建索引
  │   └─ 从《》内提取法规简称作为映射键
  │
  ├── 第2层: 法律名称检测 (_detect_law_name)
  │   └─ 短名→全名映射（"公司法"→"中华人民共和国公司法"）
  │
  └── 第3层: 机构名检测 (_detect_authority)
      └─ 地域简称→机构全称（"上海"→"上海银保监局,上海保监局"）
```

### 3.3 索引构建算法

**文件名索引 `_build_source_index()`**：
- 启动时扫描 `src/txt_files/` 和 `data/testfiles/`
- 每个文件生成两个键：完整文件名 + 《》内提取的法规简称
- 匹配时按键长度降序遍历，优先命中长键（具体法规名），避免短键（城市名）误匹配

**法律名索引 `_build_law_name_index()`**：
- 扫描文件 + 剥离 "中华人民共和国" 前缀 → 短名映射
- 硬编码补充 23 部常见法律简称（合同法、物权法、反垄断法等）

**机构名索引 `_build_authority_index()`**：
- 利用中国公文命名规则：**"关于" 之前的文字 = 发文机构**
- 提取地域关键词（上海、江苏、浙江、南通等 21 个城市）
- 同地域多机构合并（`"上海" → "上海保监局,上海银保监局"`）

### 3.4 使用策略（最新版本）

经实际测试，精确的 `source_filter` 和 `authority_filter` 在目标文档不在知识库或机构名不匹配时会导致 0 召回。最新版本已改为**软过滤**：

```python
# source 不再用作硬过滤，改为关键词注入查询
detected_source = self._detect_source(question)
if detected_source:
    expanded_query = f"{expanded_query} {detected_source}"

# authority/law_name 过滤结果为空时自动回退
if not chunks and (law_name_filter or authority_filter):
    chunks = self.retriever.retrieve(...无过滤...)
```

### 3.5 代码位置

| 函数 | 功能 |
|------|------|
| `_build_source_index()` | 文件名索引构建 |
| `_build_law_name_index()` | 法律名索引构建 |
| `_build_authority_index()` | 机构名索引构建 |
| `_detect_source()` | 文件名检测 |
| `_detect_law_name()` | 法律名检测 |
| `_detect_authority()` | 机构名检测 |
| `RAGChain.query()` | 实体检测调用 + 回退逻辑 |

---

## 4. 查询重写 (`rewriter.py`)

### 4.1 设计动机

用户口语问题含大量噪声词（"我想了解一下""有什么""怎么弄的"），向量检索需要简洁、实体密集的查询。同时，口语中的 "上海那边" 需要补全为 "上海银保监局"。

### 4.2 模型方案

采用 Qwen2.5-0.5B-Instruct + LoRA 微调：

| 维度 | 选择理由 |
|------|---------|
| 模型规模 | 0.5B — 重写是表面语言变换，不需要 7B 的法律推理能力 |
| 推理速度 | <100ms (CPU) / ~150ms (GPU)，不影响用户体验 |
| LoRA 微调 | 600 条金融领域改写数据，让通用模型学会法规查询改写模式 |
| Adapter 大小 | ~1MB，极轻量可分发 |

### 4.3 训练数据

```json
{"question": "上海市交强险与交通违法相联系的费率浮动标准从何时开始正式实施？",
 "query":    "上海市交强险与交通违法关联费率浮动标准正式实施时间"}
```

微调参数：`rank=8`, `target_modules=["q_proj", "v_proj"]`, `steps=340`

### 4.4 System Prompt

```python
QUERY_REWRITE_PROMPT = """你是一个中文检索查询改写助手。
你的任务是将用户的自然语言问题改写为一条简洁、精准、适合向量检索的查询语句。
请保留关键实体和意图，去掉无关废话，仅输出改写后的查询，不要输出额外解释或格式说明。"""
```

### 4.5 三级降级链

```
第1级: Qwen2.5-0.5B + LoRA (<100ms, 本地CPU/GPU)
    ↓ 加载失败 / 输出为空
第2级: 主 LLM (本地7B / DeepSeek API, temperature=0.0, max_tokens=64)
    ↓ LLM不可用 / API限流
第3级: 原始问题直接检索（不重写）
```

### 4.6 代码位置

| 文件/类 | 功能 |
|---------|------|
| `rewriter.py` → `QueryRewriter.__init__()` | 模型加载 + LoRA 挂载 |
| `rewriter.py` → `QueryRewriter.rewrite()` | 重写推理 |
| `rag_chain.py` → `RAGChain.rewrite_query()` | 三级降级调度 |
| `tools/train_rewriter.py` | LoRA 微调训练脚本 |
| `checkpoints/rewriter_lora/final/` | 最终 LoRA 权重 |

---

## 5. 并行检索 + 加权 RRF (`retriever.py`)

### 5.1 设计动机

原版 PR #1 的检索是串行的：向量检索 → BM25 → 术语索引。三路之间相互等待，总延迟 = 向量耗时 + BM25 耗时 + 术语耗时。改为并行后，总延迟 = max(向量耗时, BM25 耗时, 术语耗时)。

### 5.2 并行化实现

```python
# 全局线程池（避免每次检索重复创建）
_TPOOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="rag")

# 三路并行提交
pool = _get_pool()
futures = {
    pool.submit(self._vec_search, ...): "vec",
}
if self._has_es or self._has_bm25:
    futures[pool.submit(self._ft_search, ...)] = "ft"
if self._has_terms:
    futures[pool.submit(self._term_search, ...)] = "term"

# as_completed 收集结果，最快路径先返回
for fut in as_completed(futures):
    name = futures[fut]
    result = fut.result()
```

每个检索路径被拆分为独立方法（`_vec_search`, `_ft_search`, `_term_search`），供 `ThreadPoolExecutor` 调度。

### 5.3 加权 RRF 融合

```python
# 加权 RRF: score(d) = Σ weight_r / (60 + rank_r(d))
# BM25 权重 3.0 — 金融术语精确关键词匹配比语义向量更可靠
# 术语索引权重 2.0 — 英文缩写/规范术语补充召回
# 向量检索权重 1.0 — 语义兜底

recall_weights = [1.0]  # vec
if ft_candidates:
    recall_weights.append(3.0)  # ft
if term_candidates:
    recall_weights.append(2.0)  # term
candidates = _rrf_fusion(*recall_lists, top_k=k, weights=recall_weights)
```

### 5.4 代码位置

| 方法 | 功能 |
|------|------|
| `_get_pool()` | 线程池惰性初始化 |
| `_vec_search()` | 向量检索路径 |
| `_ft_search()` | 全文检索路径 (ES → BM25 回退) |
| `_term_search()` | 术语倒排索引路径 |
| `retrieve()` | 并行调度 + 加权 RRF 融合 + Reranker 精排 |
| `_rrf_fusion()` | 支持 weights 参数的加权 RRF 算法 |

---

## 6. 流式输出 (`llm.py` + `rag_chain.py` + `api_app.py` + `app.py`)

### 6.1 设计动机

原版 LLM 生成完成后才一次性返回，用户需等 5-15 秒看到结果。改为流式后，首个 token 出来后立即显示，感知延迟大幅降低。

### 6.2 LLM 流式生成

`llm.py` → `LocalLLM.generate_stream()`：

```python
def generate_stream(self, messages, ...):
    inputs = self._prepare_inputs(messages)
    streamer = TextIteratorStreamer(
        self.tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = {**inputs, ..., streamer=streamer}
    thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
    thread.start()
    yield from streamer  # token-by-token
```

使用 HuggingFace `TextIteratorStreamer` + 后台线程。对 API 后端（DeepSeek/Qwen）自动回退非流式。

### 6.3 RAG 链路流式封装

`rag_chain.py` → `RAGChain.query_stream()`：

```python
# 每行一个 JSON 事件，SSE 兼容格式：
{"type": "meta", "rewritten_query": "...", "sources": [...]}  # 检索元信息
{"type": "token", "text": "根"}                                # 逐 token
{"type": "token", "text": "据"}
...
{"type": "done", "confidence": {...}}                          # 完成
```

### 6.4 SSE 端点

`api_app.py` → `POST /api/qa/stream`：
```python
return StreamingResponse(
    _generate(),
    media_type="text/event-stream",
    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
)
```

### 6.5 前端流式渲染

`app.py`：
```python
answer_box = st.empty()
for raw_line in response.iter_lines(decode_unicode=True):
    event = json.loads(line)
    if event["type"] == "token":
        answer += event["text"]
        answer_box.markdown(answer + "▌")  # 逐字更新
```

### 6.6 代码位置

| 文件 | 方法/端点 | 功能 |
|------|---------|------|
| `llm.py` | `LocalLLM.generate_stream()` | 本地 LLM token-by-token 生成 |
| `rag_chain.py` | `RAGChain.query_stream()` | RAG 全流程流式封装 |
| `api_app.py` | `POST /api/qa/stream` | SSE 服务端点 |
| `app.py` | 问答输入处理 | 前端逐字渲染 |

---

## 7. 前端多文件上传 (`app.py`)

### 7.1 设计动机

原版 PR #1 把三个上传区合并成一个下拉框+单文件上传器，体验不如三个独立区域直观。

### 7.2 实现

```python
# 三个独立上传区，各含 accept_multiple_files=True
law_files = st.file_uploader("上传金融法规 PDF 或 TXT（可多选）", accept_multiple_files=True)
case_files = st.file_uploader("上传案例文件（可多选）", accept_multiple_files=True)
other_files = st.file_uploader("上传其他参考资料（可多选）", accept_multiple_files=True)

# 逐文件上传+索引，进度条 + 异常容错
for i, f in enumerate(law_files):
    progress.progress((i+1)/len(law_files))
    upload_file(...) → index_file(...)
    time.sleep(1)  # 间隔 1s 避免 milvus-lite 文件锁冲突
```

### 7.3 代码位置

`app.py` 侧边栏 → "文档管理" 区域。

---

## 8. milvus-lite 文件锁重试 (`vector_store.py`)

### 8.1 问题

Windows 环境下 milvus-lite 在连续多次 flush 时出现文件锁冲突：

```
milvus_lite.exceptions.DataDirLockedError:
  [WinError 183] 当文件已存在时，无法创建该文件
```

### 8.2 解决方案

```python
@staticmethod
def _retry_operation(op, *args, max_retries=5, operation_name="", **kwargs):
    """指数退避重试: 0.5s×2^attempt (最大8s)，最多5次"""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return op(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = 0.5 * (2 ** attempt)  # 1s, 2s, 4s, 8s
                time.sleep(wait)
    raise last_exc
```

insert 和 flush 操作均通过重试包装器执行。

### 8.3 代码位置

`vector_store.py` → `VectorStore._retry_operation()`, `_insert_batch()`, `_flush_with_retry()`

---

## 9. Windows 路径兼容

### 9.1 MILVUS_URI vs MILVUS_LOCAL_DB

**问题**：pymilvus 在 `import` 时自动读取 `MILVUS_URI` 环境变量并尝试解析为 HTTP URL。Windows 绝对路径 `C:\...\milvus_finance.db` 不是合法 URL，导致崩溃。

**解决**：将环境变量改名为 `MILVUS_LOCAL_DB`（pymilvus 不自动读取），在 `MilvusClient` 初始化时才手动传入。

```python
# vector_store.py
local_db = os.getenv("MILVUS_LOCAL_DB", "")
if local_db:
    uri = _StdPath(local_db).resolve().as_posix()  # C:\... → C:/.../
else:
    uri = f"http://{host}:{port}"
```

### 9.2 dotenv 路径加载

**问题**：`load_dotenv()` 无参调用默认从当前工作目录找 `.env`，而非从代码文件所在目录。

**解决**：所有需要读取 `.env` 的模块使用显式路径：

```python
# embedder.py / llm.py / vector_store.py
load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

# api_app.py
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
```

### 9.3 模型路径解析

**问题**：`.env` 中配置的 `RERANKER_MODEL_PATH=./models/bge-reranker-v2-m3` 是相对路径，在不同工作目录下解析不一致。

**解决**：代码中基于 `__file__` 解析为绝对路径：

```python
# embedder.py → Reranker.__init__()
_mp = Path(model_path)
if not _mp.is_absolute():
    _mp = (Path(__file__).resolve().parent.parent / model_path).resolve()

# llm.py
_mp = (Path(__file__).resolve().parent.parent / _LLM_MODEL_PATH).resolve().as_posix()
```

### 9.4 代码位置

| 文件 | 修改内容 |
|------|---------|
| `vector_store.py` | MILVUS_LOCAL_DB + as_posix() 转换 |
| `embedder.py` | load_dotenv 显式路径 + Reranker 路径解析 |
| `llm.py` | load_dotenv 显式路径 + LLM 路径解析 |
| `.env` | MILVUS_URI → MILVUS_LOCAL_DB，模型路径切换为 D 盘绝对路径 |

---

## 10. 检索过滤软回退 (`rag_chain.py`)

### 10.1 问题

经实际测试发现两个导致零召回的问题：

1. **`source_filter` 精确过滤**：检测到的文件名在知识库中不存在 → 0 召回
2. **`authority_filter` 精确过滤**：词典将 "中国保监会" 映射到 "国家金融监督管理总局"，但知识库中实际存储 "上海银保监局" 等地方机构名 → 0 召回
3. **`status_filter` 过滤**：`doc_type="other"` 的文档入库时未设置 `status` 字段 → 全部被 "有效" 过滤

### 10.2 修复

```python
# 修复1: source_filter 改为关键词注入，不做硬过滤
detected_source = self._detect_source(question)
source_filter = None
if detected_source:
    expanded_query = f"{expanded_query} {detected_source}"

# 修复2: authority/law_name 过滤 0 召回时自动回退
if not chunks and (law_name_filter or authority_filter):
    chunks = self.retriever.retrieve(..., authority_filter=None, ...)

# 修复3: status 过滤 0 召回时自动回退
if not chunks and status_filter:
    chunks = self.retriever.retrieve(..., status_filter=None, ...)
```

### 10.3 代码位置

`rag_chain.py` → `RAGChain.query()` 和 `RAGChain.query_stream()` 的检索段。

---

## 11. 工具脚本 (`tools/`)

| 脚本 | 功能 |
|------|------|
| `tools/convert_testfiles.py` | 将 `.doc`/`.docx` 文件通过 LibreOffice headless 转为 UTF-8 `.txt` |
| `tools/generate_questions.py` | 调用 DeepSeek API 从文档自动生成测试问题 |
| `tools/generate_rewrite_data.py` | 批量生成查询重写训练数据 |
| `tools/rewrite_questions_for_rag.py` | 批量重写 `questions.json` 中的问题为检索查询 |
| `tools/train_rewriter.py` | LoRA 微调查询重写小模型 |
| `tools/import_testfiles.py` | 批量导入 testfiles 到 Milvus |
| `tools/extract_dictionary.py` | 从 chunk 中自动抽取金融术语候选 |
| `tools/merge_dictionary.py` | 合并词典候选到正式词典 |

---

## 12. 关键配置

### 12.1 `.env` 核心参数

```bash
# 模型路径（已迁移至 D 盘）
EMBEDDING_MODEL_PATH=D:\rag_models\bge-small-zh-v1.5
RERANKER_MODEL_PATH=D:\rag_models\bge-reranker-v2-m3
LLM_MODEL_PATH=D:\rag_models\Qwen2.5-7B-Int4

# Milvus（不使用 MILVUS_URI 以避 pymilvus 自动解析）
MILVUS_LOCAL_DB=C:\Users\wangx\Desktop\rag_finance_system\db\milvus_finance.db
MILVUS_COLLECTION_NAME=finance_regulations

# LLM 降级
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-chat

# 检索参数
RETRIEVER_TOP_K=10
RERANKER_TOP_N=5
CHUNK_SIZE=512
CHUNK_OVERLAP=100
```

### 12.2 当前系统规模

| 指标 | 数值 |
|------|------|
| 知识库文件 | 148 份规范性文件 + 83 部法律 |
| Chunk 总数 | ~7000（含法律条文 + testfiles + 用户上传） |
| 词典覆盖 | 76 术语 + 55 法律 + 29 机构 + 500+ 别名 |
| BM25 文档数 | 6987 篇 |
| 部署方式 | RTX 5080 16GB 本地 GPU + milvus-lite 嵌入式 |

---

## 13. 开发迭代历史

按时间线排列的核心提交：

| 提交 | 功能 |
|------|------|
| `aa7fcdd` | 实体检测 + 查询重写 + other 文档类型 + Sigmoid 归一化 |
| `5cf234a` | 并行检索 + LLM 流式输出 + SSE 端点 |
| `4d20a4e` | milvus-lite 文件锁指数退避重试 |
| `31bce7a` | 三独立上传区 + MILVUS_URI→MILVUS_LOCAL_DB + dotenv 路径修复 |
| `72d2f86` | 选择性合入 fork (Reranker FP16 + 加权 RRF + 时效管理 + 词典扩充) |
| `129c99d` | 模型迁移 D 盘 + 路径全面修复 |
| `20f7765` | 检索过滤软回退（source→查询增强 + authority 0 召回回退） |

---

## 14. 启动方式

```bash
# 终端 1 — 后端
cd C:\Users\wangx\Desktop\rag_finance_system
uvicorn rag_finance_system.api_app:app --host 0.0.0.0 --port 8000

# 终端 2 — 前端
cd C:\Users\wangx\Desktop\rag_finance_system
streamlit run rag_finance_system/app.py

# 浏览器打开 http://localhost:8501
```
