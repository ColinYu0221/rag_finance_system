"""
rag_chain.py
RAG主链路：检索 → Prompt组装 → LLM生成 → 答案+溯源
"""

import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


# ========================
# 法律名称识别
# ========================

def _build_law_name_index() -> dict[str, str]:
    """扫描 txt_files 目录，构建 {短名: 全名} 映射。

    "中华人民共和国公司法" → 短名 "公司法"
    "中华人民共和国民法典"  → 短名 "民法典"
    """
    import glob as _glob

    _txt_dir = Path(__file__).resolve().parent / "txt_files"
    _mapping: dict[str, str] = {}

    for _fp in _glob.glob(str(_txt_dir / "*.txt")):
        _stem = os.path.basename(_fp)
        _stem = re.sub(r"_\d{8}\.txt$", "", _stem)  # 去日期后缀
        _mapping[_stem] = _stem

        # 剥离"中华人民共和国"生成短名
        if _stem.startswith("中华人民共和国"):
            _short = _stem[7:]  # len("中华人民共和国") = 7
            _mapping[_short] = _stem

    # 扩展常见简称映射
    _extras = {
        "公司法": "中华人民共和国公司法",
        "证券法": "中华人民共和国证券法",
        "保险法": "中华人民共和国保险法",
        "民法典": "中华人民共和国民法典",
        "刑法": "中华人民共和国刑法",
        "商业银行法": "中华人民共和国商业银行法",
        "信托法": "中华人民共和国信托法",
        "票据法": "中华人民共和国票据法",
        "海商法": "中华人民共和国海商法",
        "企业破产法": "中华人民共和国企业破产法",
        "担保法": "中华人民共和国担保法",
        "合同法": "中华人民共和国合同法",
        "物权法": "中华人民共和国物权法",
        "著作权法": "中华人民共和国著作权法",
        "专利法": "中华人民共和国专利法",
        "商标法": "中华人民共和国商标法",
        "反垄断法": "中华人民共和国反垄断法",
        "反洗钱法": "中华人民共和国反洗钱法",
        "个人信息保护法": "中华人民共和国个人信息保护法",
        "网络安全法": "中华人民共和国网络安全法",
        "数据安全法": "中华人民共和国数据安全法",
        "证券投资基金法": "中华人民共和国证券投资基金法",
        "期货和衍生品法": "中华人民共和国期货和衍生品法",
    }
    for _k, _v in _extras.items():
        _mapping[_k] = _v

    return _mapping


_LAW_NAME_INDEX = _build_law_name_index()


def _build_source_index() -> dict[str, str]:
    """扫描所有 txt 文件，构建 {文件名/法规简称: 完整source} 索引。

    从文件名中提取《》内的法规简称（10-60 字），同时保留完整文件名作为键。
    "上海银行业金融机构防范非法集资工作机制（暂行）"
        → "上海银保监局办公室关于印发《...》的通知"
    """
    import glob as _glob

    _mapping: dict[str, str] = {}

    _src_dirs = [
        Path(__file__).resolve().parent / "txt_files",
        Path(__file__).resolve().parent.parent.parent / "data" / "testfiles",
    ]

    for _src_dir in _src_dirs:
        if not _src_dir.exists():
            continue
        for _fp in _glob.glob(str(_src_dir / "**" / "*.txt"), recursive=True):
            _title = os.path.basename(_fp).replace(".txt", "")
            _title = re.sub(r"_\d{8}$", "", _title)
            _title = re.sub(r"^\d{1,2}\.", "", _title).strip()

            # 完整文件名 → 自身
            _mapping[_title] = _title

            # 提取《》内的法规名称作为短键
            _regulations = re.findall(r"《([^》]+)》", _title)
            for _reg in _regulations:
                _reg = _reg.strip()
                if 4 <= len(_reg) <= 80 and _reg not in _mapping:
                    _mapping[_reg] = _title

    return _mapping


_SOURCE_INDEX = _build_source_index()


def _build_authority_index() -> dict[str, str]:
    """扫描 txt_files 和 testfiles 目录，构建 {短名: 全称} 机构索引。

    "上海" → "上海银保监局"
    "南通" → "南通监管分局"
    """
    import glob as _glob

    _mapping: dict[str, str] = {}
    _seen: set[str] = set()

    _src_dirs = [
        Path(__file__).resolve().parent / "txt_files",
        Path(__file__).resolve().parent.parent.parent / "data" / "testfiles",
    ]

    for _src_dir in _src_dirs:
        if not _src_dir.exists():
            continue
        for _fp in _glob.glob(str(_src_dir / "**" / "*.txt"), recursive=True):
            _title = os.path.basename(_fp).replace(".txt", "")
            _title = re.sub(r"_\d{8}$", "", _title)
            # 清理文件名前导数字编号（如 "1." "10." "20."）
            _title = re.sub(r"^\d{1,2}\.", "", _title).strip()

            # 提取 "关于" 前的机构名
            for _kw in ["关于", "办公室关于"]:
                if _kw in _title:
                    _auth = _title.split(_kw)[0].strip()
                    _auth = re.sub(r"^(国家金融监督管理总局|中国银保监会|中国保监会|中国银监会)", "", _auth)
                    _auth = re.sub(r"(办公室|秘书处)$", "", _auth)
                    if len(_auth) >= 4 and _auth not in _seen:
                        _seen.add(_auth)
                        _mapping[_auth] = _auth
                        # 提取地域关键词作为短名，一个地域可能对应多个机构
                        for _city in ["上海", "江苏", "浙江", "南通", "泰州", "淮安", "宿迁",
                                       "盐城", "苏州", "连云港", "金华", "丽水", "舟山", "宁波",
                                       "杭州", "温州", "嘉兴", "湖州", "绍兴", "台州", "衢州"]:
                            if _city in _auth:
                                _existing = _mapping.get(_city)
                                if _existing and _existing != _auth:
                                    # 同一地域多个机构 → 合并为逗号分隔
                                    _mapping[_city] = _existing + "," + _auth
                                else:
                                    _mapping[_city] = _auth
                    break

    return _mapping


_AUTHORITY_INDEX = _build_authority_index()

# System Prompt：约束LLM只基于上下文回答，防止幻觉
SYSTEM_PROMPT = """你是一个专业的金融法规知识助手。你的任务是根据提供的金融制度条文，准确回答用户的问题。

**重要规则**：
1. 只基于【参考条文】中的内容回答，不要凭空推测或引用参考条文之外的知识
2. 如果参考条文中没有相关内容，请明确告知"暂未找到相关条文，建议查阅完整法规原文"
3. 回答时请引用具体的条文来源（文件名、条文编号）
4. 保持回答专业、简洁、准确

回答格式：
- 先给出直接答案
- 再列出相关条文（格式：【来源：文件名 条文编号】）"""

QUERY_REWRITE_SYSTEM_PROMPT = """你是一个中文检索查询改写助手。
你的任务是将用户的自然语言问题改写为一条简洁、精准、适合向量检索的查询语句。
请保留关键实体和意图，去掉无关废话，仅输出改写后的查询，不要输出额外解释或格式说明。"""


def build_prompt(query: str, chunks: List[Dict[str, Any]], graph_facts: Optional[List[str]] = None) -> List[dict]:
    """
    组装Prompt消息列表
    Args:
        query: 用户问题
        chunks: 检索到的相关chunks
        graph_facts: 图谱补充事实
    Returns:
        messages列表（适配OpenAI格式）
    """
    if not chunks:
        context = "（未检索到相关条文）"
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "未知文件")
            article = chunk.get("article_num", "")
            doc_type = chunk.get("doc_type", "law")
            if doc_type == "case":
                prefix = "案例"
            elif doc_type == "other":
                prefix = "其它资料"
            else:
                prefix = "法规"
            source_tag = f"【{prefix} {source}{'  ' + article if article else ''}】"
            context_parts.append(f"{i}. {source_tag}\n{chunk['text']}")
        context = "\n\n".join(context_parts)

    graph_context = ""
    if graph_facts:
        graph_context = "\n\n【图谱补充关系】\n" + "\n".join(
            f"- {fact}" for fact in graph_facts
        )

    user_message = f"""【参考条文】
{context}{graph_context}

【用户问题】
{query}

请根据以上参考条文回答问题："""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


class RAGChain:
    """
    RAG主链路
    整合：Retriever（检索） + LLM（生成） + FinanceDictionary（实体识别） + KnowledgeGraph（图谱扩展）
    """

    def __init__(self, retriever, llm, rewriter=None, dictionary=None, knowledge_graph=None):
        self.retriever = retriever
        self.llm = llm
        self.rewriter = rewriter
        self.dictionary = dictionary
        self.kg = knowledge_graph

    @staticmethod
    def _detect_source(question: str) -> Optional[str]:
        """从问题中识别提及的具体文件名/法规简称，返回完整 source。

        "上海银行业金融机构防范非法集资工作机制"  →  "上海银保监局办公室关于印发《...》的通知"
        优先匹配最长的键（避免"上海"被误匹配为机构而不是文件名的一部分）。
        """
        for short in sorted(_SOURCE_INDEX, key=len, reverse=True):
            if len(short) >= 6 and short in question:
                full = _SOURCE_INDEX[short]
                logger.info(f"检测到文件名: {short[:60]} → {full[:60]}")
                return full
        return None

    def _detect_law_name(self, question: str) -> Optional[str]:
        """优先使用词典检测法律名称，回退到文件名索引。"""
        if self.dictionary:
            entities = self.dictionary.detect_entities(question)
            if entities["law_names"]:
                law = entities["law_names"][0]
                logger.info(f"词典检测法律: {law}")
                return law
        # 回退：旧文件名索引
        for short, full in _LAW_NAME_INDEX.items():
            if len(short) >= 3 and short in question:
                logger.info(f"索引检测法律: {short} → {full}")
                return full
        return None

    def _detect_authority(self, question: str) -> Optional[str]:
        """优先使用词典检测监管机构，回退到文件名索引。"""
        if self.dictionary:
            entities = self.dictionary.detect_entities(question)
            if entities["authorities"]:
                auth = ",".join(entities["authorities"])
                logger.info(f"词典检测机构: {auth}")
                return auth
        # 回退：旧文件名索引
        for short, full in _AUTHORITY_INDEX.items():
            if len(short) >= 2 and short in question:
                logger.info(f"索引检测机构: {short} → {full}")
                return full
        return None

    def rewrite_query(self, question: str) -> str:
        """将用户问题改写为适合检索的查询语句（优先用小模型，回退大模型）。"""
        if self.rewriter:
            try:
                rewritten = self.rewriter.rewrite(question).strip()
                if rewritten and len(rewritten) >= 2:
                    return rewritten
            except Exception as e:
                logger.warning(f"小模型查询重写失败: {e}，回退大模型")

        messages = [
            {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"请将以下问题改写为适合向量检索的简洁查询：\n\n{question}\n\n只输出改写后的查询。"
            },
        ]
        rewritten = self.llm.generate(messages, max_new_tokens=64, temperature=0.0).strip()
        return rewritten or question

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        use_query_rewrite: bool = True,
        use_query_expansion: bool = True,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        max_new_tokens: int = 1024,
        include_historical: bool = False,
    ) -> Dict[str, Any]:
        """
        完整问答流程
        Returns:
            {
                "question": str,
                "answer": str,
                "sources": [{"source": str, "article_num": str, "text": str, "score": float}],
                "confidence": {"total": float, "retrieval": float, "coverage": float},
            }
        """
        logger.info(f"问答请求: {question[:60]}...")

        # 0. 词典实体检测 + 查询扩展
        law_name_filter = None
        authority_filter = None
        expanded_query = question
        entities: dict[str, list[str]] = {"terms": [], "law_names": [], "authorities": []}

        if self.dictionary:
            entities = self.dictionary.detect_entities(question)
            if entities["law_names"]:
                law_name_filter = entities["law_names"][0]
            if entities["authorities"]:
                authority_filter = ",".join(entities["authorities"])
            if use_query_expansion and entities["terms"]:
                expanded_query = self.dictionary.expand_query(question)

        # 0b. 文件名检测 → 不再用作硬过滤，而是追加到查询中增强检索
        # 原因：精确 source_filter 命中一个知识库中不存在的文件 → 0 召回
        # 改为把文件名关键词注入 expanded_query，向量/BM25 自动找内容最相关的文档
        detected_source = self._detect_source(question)
        source_filter = None  # 不再使用硬过滤

        if detected_source:
            # 把检测到的文件名作为额外关键词加入查询
            expanded_query = f"{expanded_query} {detected_source}"
            logger.info(f"检测到文件名关键词，已注入查询: {detected_source[:60]}...")

        # 词典未命中时回退旧索引
        if not law_name_filter:
            law_name_filter = self._detect_law_name(question)
        if not authority_filter:
            authority_filter = self._detect_authority(question)

        # 1. 查询重写（对扩展后的查询做重写，保留别名提升召回）
        rewritten_query = expanded_query
        if use_query_rewrite:
            rewritten_query = self.rewrite_query(expanded_query)
            logger.info(f"重写查询: {rewritten_query[:80]}...")

        # 1b. 知识图谱扩展召回
        graph_articles: list[dict] = []
        graph_facts: list[str] = []
        if self.kg and self.kg._connected:
            try:
                authority_list = entities.get("authorities") if self.dictionary else None
                graph_articles = self.kg.get_related_articles(
                    law_names=[law_name_filter] if law_name_filter else None,
                    terms=entities.get("terms") if self.dictionary else None,
                    authorities=authority_list,
                    max_results=5,
                )
                graph_facts = self.kg.get_graph_facts(
                    law_names=[law_name_filter] if law_name_filter else None,
                    terms=entities.get("terms") if self.dictionary else None,
                    authorities=authority_list,
                    max_results=6,
                )
                if graph_articles:
                    logger.info(f"图谱扩展召回 {len(graph_articles)} 条关联条文")
                if graph_facts:
                    logger.info(f"图谱补充关系 {len(graph_facts)} 条")
            except Exception as e:
                logger.warning(f"图谱扩展失败: {e}")

        # 2. 检索（默认只看有效版本，0 结果时回退不过滤 status）
        status_filter = None if include_historical else "有效"
        chunks = self.retriever.retrieve(
            query=rewritten_query,
            top_k=top_k,
            use_reranker=use_reranker,
            source_filter=source_filter,
            doc_type_filter=doc_type_filter,
            law_name_filter=law_name_filter,
            authority_filter=authority_filter,
            status_filter=status_filter,
        )
        if not chunks and status_filter:
            # status 过滤导致 0 召回（例如 other 类文档无 status 字段）→ 去掉过滤重试
            logger.warning(f"status_filter={status_filter} 无结果，回退无 status 过滤")
            status_filter = None
            chunks = self.retriever.retrieve(
                query=rewritten_query, top_k=top_k, use_reranker=use_reranker,
                source_filter=None, doc_type_filter=doc_type_filter,
                law_name_filter=None, authority_filter=None, status_filter=None,
            )
        if not chunks and (law_name_filter or authority_filter):
            # 内容过滤导致 0 召回（如词典检测到"国家金融监督管理总局"但库内是"上海银保监局"）
            logger.warning(f"内容过滤导致 0 召回，回退无过滤检索")
            chunks = self.retriever.retrieve(
                query=rewritten_query, top_k=top_k, use_reranker=use_reranker,
                source_filter=None, doc_type_filter=doc_type_filter,
                law_name_filter=None, authority_filter=None, status_filter=None,
            )
        # 2b. 合并图谱扩展结果到检索结果后（图谱结果排在后，不参与reranker评分）
        seen_chunk_ids = {c.get("chunk_id", "") for c in chunks}
        for ga in graph_articles:
            gid = ga.get("chunk_id", "")
            if gid and gid not in seen_chunk_ids:
                seen_chunk_ids.add(gid)
                chunks.append({
                    "text": ga.get("text", ""),
                    "source": ga.get("source", ""),
                    "article_num": ga.get("article_num", ""),
                    "doc_type": "law",
                    "law_name": ga.get("law_name", ""),
                    "score": 0.0,
                    "_graph_relation": ga.get("relation", ""),
                })

        # 3. 构建Prompt
        messages = build_prompt(question, chunks, graph_facts=graph_facts)

        # 3. LLM生成
        answer = self.llm.generate(messages, max_new_tokens=max_new_tokens)
        logger.info(f"生成答案长度: {len(answer)} 字符")

        # 4. 整理溯源信息
        sources = [
            {
                "source": c.get("source", ""),
                "article_num": c.get("article_num", ""),
                "text": c.get("text", "")[:300],  # 前端展示截断
                "score": round(c.get("reranker_score", c.get("score", 0.0)), 4),
            }
            for c in chunks
        ]

        # 5. 计算可信度
        confidence = self.retriever.compute_confidence(question, answer, chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "rewritten_query": rewritten_query,
        }

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
        use_query_rewrite: bool = True,
        use_query_expansion: bool = True,
        source_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        max_new_tokens: int = 1024,
        include_historical: bool = False,
    ):
        """流式问答：检索后逐 token 生成，同时收集完整回答用于溯源。

        Yields:
            每行 JSON: {"type": "token"|"meta"|"done", ...}
        """
        import json as _json

        # 0-3: 检索（与 query() 相同）
        logger.info(f"流式问答请求: {question[:60]}...")

        law_name_filter = None
        authority_filter = None
        expanded_query = question
        entities: dict[str, list[str]] = {"terms": [], "law_names": [], "authorities": []}

        if self.dictionary:
            entities = self.dictionary.detect_entities(question)
            if entities["law_names"]:
                law_name_filter = entities["law_names"][0]
            if entities["authorities"]:
                authority_filter = ",".join(entities["authorities"])
            if use_query_expansion and entities["terms"]:
                expanded_query = self.dictionary.expand_query(question)

        detected_source = self._detect_source(question)
        source_filter = None  # 硬过滤改为查询增强

        if detected_source:
            expanded_query = f"{expanded_query} {detected_source}"

        if not law_name_filter:
            law_name_filter = self._detect_law_name(question)
        if not authority_filter:
            authority_filter = self._detect_authority(question)

        rewritten_query = expanded_query
        if use_query_rewrite:
            rewritten_query = self.rewrite_query(expanded_query)

        # 知识图谱
        graph_articles: list[dict] = []
        graph_facts: list[str] = []
        if self.kg and self.kg._connected:
            try:
                authority_list = entities.get("authorities") if self.dictionary else None
                graph_articles = self.kg.get_related_articles(
                    law_names=[law_name_filter] if law_name_filter else None,
                    terms=entities.get("terms") if self.dictionary else None,
                    authorities=authority_list, max_results=5,
                )
                graph_facts = self.kg.get_graph_facts(
                    law_names=[law_name_filter] if law_name_filter else None,
                    terms=entities.get("terms") if self.dictionary else None,
                    authorities=authority_list, max_results=6,
                )
            except Exception as e:
                logger.warning(f"图谱扩展失败: {e}")

        status_filter = None if include_historical else "有效"
        chunks = self.retriever.retrieve(
            query=rewritten_query, top_k=top_k, use_reranker=use_reranker,
            source_filter=source_filter, doc_type_filter=doc_type_filter,
            law_name_filter=law_name_filter, authority_filter=authority_filter,
            status_filter=status_filter,
        )
        if not chunks and status_filter:
            status_filter = None
            chunks = self.retriever.retrieve(
                query=rewritten_query, top_k=top_k, use_reranker=use_reranker,
                source_filter=None, doc_type_filter=doc_type_filter,
                law_name_filter=None, authority_filter=None, status_filter=None,
            )
        if not chunks and (law_name_filter or authority_filter):
            chunks = self.retriever.retrieve(
                query=rewritten_query, top_k=top_k, use_reranker=use_reranker,
                source_filter=None, doc_type_filter=doc_type_filter,
                law_name_filter=None, authority_filter=None, status_filter=None,
            )

        # 图谱条文补充到上下文
        if graph_articles:
            for ga in graph_articles:
                chunks.append({"text": ga.get("text", ""), "source": ga.get("source", "图谱关联"),
                               "article_num": ga.get("article_num", ""), "score": 0.5})

        messages = build_prompt(expanded_query, chunks, graph_facts=graph_facts)

        # 发送元信息（重写查询 + sources）
        sources_preview = [
            {"source": c.get("source", ""), "article_num": c.get("article_num", ""),
             "text": c.get("text", "")[:200], "score": round(c.get("score", 0.0), 4)}
            for c in chunks[:self.retriever.reranker_top_n]
        ]
        yield _json.dumps({"type": "meta", "rewritten_query": rewritten_query,
                           "sources": sources_preview}, ensure_ascii=False) + "\n"

        # 流式生成
        full_answer = ""
        try:
            for token in self.llm.generate_stream(messages, max_new_tokens=max_new_tokens):
                full_answer += token
                yield _json.dumps({"type": "token", "text": token}, ensure_ascii=False) + "\n"
        except AttributeError:
            # 回退：LLM 不支持流式 → 一次性生成
            full_answer = self.llm.generate(messages, max_new_tokens=max_new_tokens)
            yield _json.dumps({"type": "token", "text": full_answer}, ensure_ascii=False) + "\n"

        # 可信度
        confidence = self.retriever.compute_confidence(question, full_answer, chunks)
        yield _json.dumps({"type": "done", "confidence": confidence}, ensure_ascii=False) + "\n"
