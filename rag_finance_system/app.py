"""
app.py
Streamlit 前端 — 金融制度 RAG 问答系统
纯 HTTP 客户端，所有后端调用走 FastAPI。
"""

import os
import time
import requests
import streamlit as st
from requests.exceptions import ConnectionError, Timeout, RequestException

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")

# ===== 页面配置 =====
st.set_page_config(
    page_title="金融制度知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== Session State 初始化 =====
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None


# ===== HTTP Helper 函数 =====

def check_api_health(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/openapi.json", timeout=3)
        return r.status_code == 200
    except (ConnectionError, Timeout, RequestException):
        return False


def upload_file(api_base: str, file_bytes: bytes, filename: str, doc_type: str) -> dict:
    resp = requests.post(
        f"{api_base}/api/documents/upload",
        files={"file": (filename, file_bytes)},
        data={"doc_type": doc_type},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def index_file(api_base: str, file_path: str, doc_type: str) -> dict:
    resp = requests.post(
        f"{api_base}/api/documents/index",
        json={"file_path": file_path, "doc_type": doc_type},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def ask_question(
    api_base: str,
    question: str,
    doc_type_filter: str | None,
    use_reranker: bool,
    use_query_rewrite: bool,
) -> dict:
    payload: dict = {
        "question": question,
        "use_reranker": use_reranker,
        "use_query_rewrite": use_query_rewrite,
    }
    if doc_type_filter:
        payload["doc_type_filter"] = doc_type_filter
    resp = requests.post(
        f"{api_base}/api/qa",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def _handle_api_error(exc: RequestException) -> str:
    if hasattr(exc, "response") and exc.response is not None:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = exc.response.text or str(exc)
        return f"API错误 {exc.response.status_code}: {detail}"
    if isinstance(exc, ConnectionError):
        return "无法连接到API服务器，请确认FastAPI已启动"
    if isinstance(exc, Timeout):
        return "请求超时，服务器处理时间过长"
    return f"请求失败: {exc}"


# ===== 渲染 Helper =====

def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📎 查看溯源条文 ({len(sources)} 条)"):
        for i, src in enumerate(sources, 1):
            conf_color = "green" if src["score"] > 0.7 else "orange" if src["score"] > 0.4 else "red"
            st.markdown(
                f"**[{i}] 【{src['source']} {src['article_num']}】** "
                f":{conf_color}[相关度 {src['score']:.2%}]"
            )
            st.text(src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"])
            if i < len(sources):
                st.divider()


def render_confidence(conf: dict):
    col1, col2, col3 = st.columns(3)
    col1.metric("综合可信度", f"{conf['total']:.1%}")
    col2.metric("检索相关性", f"{conf['retrieval']:.1%}")
    col3.metric("答案覆盖度", f"{conf['coverage']:.1%}")


# ===== 侧边栏 =====
with st.sidebar:
    st.title("⚙️ 系统设置")

    # API URL 配置
    url_input = st.text_input(
        "API服务器URL",
        value=st.session_state.api_base_url,
        key="api_url_input",
        help="FastAPI 后端地址，默认 http://localhost:8000",
    )
    if url_input.strip() != st.session_state.api_base_url:
        st.session_state.api_base_url = url_input.strip()
        st.session_state.api_ok = None  # 重置健康状态

    # 健康检查
    api_base = st.session_state.api_base_url
    st.session_state.api_ok = check_api_health(api_base)
    if st.session_state.api_ok:
        st.success("API 服务正常", icon="✅")
    else:
        st.warning("API 服务未响应，请启动 FastAPI 后刷新", icon="⚠️")

    st.divider()

    # 检索设置
    use_reranker = st.toggle("启用 Reranker 精排", value=True,
                             help="提升检索精度，需要服务端加载 Reranker 模型")
    use_query_rewrite = st.toggle("启用查询重写", value=True,
                                  help="将用户问题改写为更适合向量检索的查询")
    mode = st.selectbox(
        "检索模式",
        ["全部", "仅法条", "仅案例", "仅其他"],
        index=0,
        help="限定本次问答的检索范围",
    )

    st.divider()
    st.subheader("文档管理")

    # ── 上传法条 ──
    law_files = st.file_uploader(
        "上传金融法规 PDF 或 TXT（可多选）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="law_uploader",
        help="支持中文 PDF 或纯文本 TXT，最大 200MB，可批量选择多个文件",
    )
    if law_files:
        if st.button("解析法条并建立索引", type="primary", key="law_btn", disabled=not st.session_state.api_ok):
            total = 0
            errors = 0
            progress = st.progress(0, text=f"0/{len(law_files)}")
            for i, f in enumerate(law_files):
                try:
                    up = upload_file(api_base, f.getvalue(), f.name, "law")
                    ix = index_file(api_base, up["file_path"], "law")
                    total += ix["chunk_count"]
                    time.sleep(1)  # 避免 milvus-lite 文件锁冲突
                except RequestException as e:
                    st.warning(f"跳过 {f.name}: {_handle_api_error(e)}")
                    errors += 1
                progress.progress((i + 1) / len(law_files), text=f"{i + 1}/{len(law_files)}  ({total} chunks)")
            st.success(f"✅ 法条入库完成：{len(law_files) - errors} 个文件，{total} 条")

    # ── 上传案例 ──
    case_files = st.file_uploader(
        "上传案例文件（可多选）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="case_uploader",
        help="支持裁判文书 PDF 或纯文本 TXT，最大 200MB，可批量选择多个文件",
    )
    if case_files:
        if st.button("解析案例并建立索引", type="primary", key="case_btn", disabled=not st.session_state.api_ok):
            total = 0
            errors = 0
            progress = st.progress(0, text=f"0/{len(case_files)}")
            for i, f in enumerate(case_files):
                try:
                    up = upload_file(api_base, f.getvalue(), f.name, "case")
                    ix = index_file(api_base, up["file_path"], "case")
                    total += ix["chunk_count"]
                    time.sleep(1)  # 避免 milvus-lite 文件锁冲突
                except RequestException as e:
                    st.warning(f"跳过 {f.name}: {_handle_api_error(e)}")
                    errors += 1
                progress.progress((i + 1) / len(case_files), text=f"{i + 1}/{len(case_files)}  ({total} chunks)")
            st.success(f"✅ 案例入库完成：{len(case_files) - errors} 个文件，{total} 条")

    # ── 上传其他资料 ──
    other_files = st.file_uploader(
        "上传其他参考资料（可多选）",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key="other_uploader",
        help="支持 PDF 或 TXT，如学术文献、研究报告、政策解读等，最大 200MB，可批量选择多个文件",
    )
    if other_files:
        if st.button("解析资料并建立索引", type="primary", key="other_btn", disabled=not st.session_state.api_ok):
            total = 0
            errors = 0
            progress = st.progress(0, text=f"0/{len(other_files)}")
            for i, f in enumerate(other_files):
                try:
                    up = upload_file(api_base, f.getvalue(), f.name, "other")
                    ix = index_file(api_base, up["file_path"], "other")
                    total += ix["chunk_count"]
                    time.sleep(1)  # 避免 milvus-lite 文件锁冲突
                except RequestException as e:
                    st.warning(f"跳过 {f.name}: {_handle_api_error(e)}")
                    errors += 1
                progress.progress((i + 1) / len(other_files), text=f"{i + 1}/{len(other_files)}  ({total} chunks)")
            st.success(f"✅ 资料入库完成：{len(other_files) - errors} 个文件，{total} 条")

    st.divider()
    st.subheader("批量导入")
    if st.button(
        "一键导入 data/testfiles 中的 TXT 文件",
        type="secondary",
        key="batch_import",
        disabled=not st.session_state.api_ok,
    ):
        import glob as _glob
        from pathlib import Path as _Path
        _txt_files = sorted(_glob.glob(str(_Path("data/testfiles") / "**" / "*.txt"), recursive=True))
        if not _txt_files:
            st.warning("未找到 .txt 文件，请先运行 tools/convert_testfiles.py 转换")
        else:
            progress = st.progress(0, text=f"0/{len(_txt_files)}")
            total_chunks = 0
            errors = 0
            for i, fp in enumerate(_txt_files):
                fname = _Path(fp).name
                try:
                    with open(fp, "rb") as fh:
                        up = upload_file(api_base, fh.read(), fname, "other")
                    ix = index_file(api_base, up["file_path"], "other")
                    total_chunks += ix["chunk_count"]
                except Exception as e:
                    st.warning(f"跳过 {fname}: {e}")
                    errors += 1
                progress.progress(
                    (i + 1) / len(_txt_files),
                    text=f"{i + 1}/{len(_txt_files)}  ({total_chunks} chunks)",
                )
            st.success(f"批量导入完成！{len(_txt_files) - errors} 个文件，共 {total_chunks} 个 chunk")

    st.divider()
    st.caption("金融制度 RAG 问答系统\n向量检索 + Reranker + LLM")


# ===== 主界面 =====
st.title("📚 金融制度知识问答系统")
st.caption("基于 RAG 的金融法规智能问答 | bge-small-zh-v1.5 + Qwen2.5")

# 检索模式 → doc_type_filter
_mode_map = {"全部": None, "仅法条": "law", "仅案例": "case", "仅其他": "other"}
doc_type_filter = _mode_map[mode]

# 历史消息回放
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("question"):
            st.caption(f"已改写查询：{msg['rewritten_query']}")
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("confidence"):
            render_confidence(msg["confidence"])

# 用户输入
if question := st.chat_input(
    "请输入您的金融法规问题...",
    disabled=not st.session_state.api_ok,
):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        result = None
        with st.spinner("检索相关条文并生成答案..."):
            try:
                result = ask_question(
                    api_base,
                    question,
                    doc_type_filter,
                    use_reranker,
                    use_query_rewrite,
                )
            except RequestException as e:
                err_msg = _handle_api_error(e)
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})

        if result is not None:
            answer = result["answer"]
            rewritten = result.get("rewritten_query")
            sources = result.get("sources", [])
            confidence = result.get("confidence", {})

            st.markdown(answer)
            if rewritten and rewritten != question:
                st.caption(f"已改写查询：{rewritten}")
            render_sources(sources)
            if confidence:
                render_confidence(confidence)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "question": question,
                "rewritten_query": rewritten,
                "sources": sources,
                "confidence": confidence,
            })
