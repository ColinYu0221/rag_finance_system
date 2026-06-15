"""
app.py
Streamlit 前端 — 金融制度 RAG 问答系统
纯 HTTP 客户端，所有后端调用走 FastAPI。
"""

import json
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
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "conversation_list" not in st.session_state:
    st.session_state.conversation_list = []
if "mysql_ok" not in st.session_state:
    st.session_state.mysql_ok = None
if "last_health_check" not in st.session_state:
    st.session_state.last_health_check = 0
if "favorites_list" not in st.session_state:
    st.session_state.favorites_list = None
if "last_fav_fetch" not in st.session_state:
    st.session_state.last_fav_fetch = 0


# ===== HTTP Helper 函数 =====

HEALTH_CHECK_INTERVAL = 30  # 秒

def check_api_health(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/openapi.json", timeout=3)
        return r.status_code == 200
    except (ConnectionError, Timeout, RequestException):
        return False


def check_mysql(api_base: str) -> bool:
    try:
        r = requests.get(f"{api_base}/api/conversations", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def cached_health_check(api_base: str) -> tuple[bool, bool]:
    """带缓存的健康检查，30秒内不重复请求。返回 (api_ok, mysql_ok)。"""
    now = time.time()
    if now - st.session_state.last_health_check < HEALTH_CHECK_INTERVAL:
        return st.session_state.api_ok, st.session_state.mysql_ok

    api_ok = check_api_health(api_base)
    mysql_ok = False
    if api_ok:
        mysql_ok = check_mysql(api_base)

    st.session_state.api_ok = api_ok
    st.session_state.mysql_ok = mysql_ok
    st.session_state.last_health_check = now
    return api_ok, mysql_ok


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


# ===== 对话历史 API =====

def fetch_conversations(api_base: str) -> list[dict]:
    try:
        r = requests.get(f"{api_base}/api/conversations", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def create_conversation(api_base: str, title: str = "新对话") -> dict | None:
    try:
        r = requests.post(f"{api_base}/api/conversations", json={"title": title}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_conversation_detail(api_base: str, conv_id: str) -> dict | None:
    try:
        r = requests.get(f"{api_base}/api/conversations/{conv_id}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def delete_conversation_api(api_base: str, conv_id: str) -> bool:
    try:
        r = requests.delete(f"{api_base}/api/conversations/{conv_id}", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False


# ===== 收藏 API =====

def add_favorite(api_base: str, fav_type: str, conversation_id: str | None = None,
                 message_id: str | None = None, source_data: dict | None = None) -> dict | None:
    try:
        payload: dict = {"fav_type": fav_type}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if message_id:
            payload["message_id"] = message_id
        if source_data:
            payload["source_data"] = source_data
        r = requests.post(f"{api_base}/api/favorites", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_favorites(api_base: str, fav_type: str | None = None) -> list[dict]:
    try:
        params = {}
        if fav_type:
            params["fav_type"] = fav_type
        r = requests.get(f"{api_base}/api/favorites", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def delete_favorite(api_base: str, fav_id: str) -> bool:
    try:
        r = requests.delete(f"{api_base}/api/favorites/{fav_id}", timeout=5)
        r.raise_for_status()
        return True
    except Exception:
        return False


# ===== 渲染 Helper =====

def render_sources(sources: list[dict], msg_idx: int = 0, conv_id: str | None = None):
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
            # 收藏单条条文按钮
            if st.session_state.mysql_ok:
                col_spacer, col_btn = st.columns([8, 1])
                with col_btn:
                    if st.button("⭐", key=f"fav_src_{msg_idx}_{i}",
                                 help="收藏此条文"):
                        result = add_favorite(
                            st.session_state.api_base_url,
                            fav_type="source",
                            conversation_id=conv_id,
                            source_data=src,
                        )
                        if result:
                            st.toast("已收藏", icon="⭐")
                            st.session_state.favorites_list = None
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
        st.session_state.api_ok = None
        st.session_state.mysql_ok = None

    # 健康检查（30秒缓存）
    api_base = st.session_state.api_base_url
    api_ok, mysql_ok = cached_health_check(api_base)
    if api_ok:
        st.success("API 服务正常", icon="✅")
    else:
        st.warning("API 服务未响应，请启动 FastAPI 后刷新", icon="⚠️")

    if api_ok:
        if mysql_ok:
            st.success("MySQL 连接正常", icon="✅")
        else:
            st.info("MySQL 未连接，对话历史/收藏功能不可用", icon="ℹ️")

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

    # ── 对话历史 ──
    if st.session_state.mysql_ok:
        st.divider()
        st.subheader("💬 对话历史")

        col_new, col_refresh = st.columns([2, 1])
        with col_new:
            if st.button("📝 新对话", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_id = None
        with col_refresh:
            if st.button("🔄", use_container_width=True, help="刷新列表"):
                st.session_state.conversation_list = fetch_conversations(api_base)
                st.session_state.favorites_list = None

        # 懒加载对话列表（仅首次和刷新时拉取）
        if not st.session_state.conversation_list:
            st.session_state.conversation_list = fetch_conversations(api_base)

        for conv in st.session_state.conversation_list:
            col_title, col_del = st.columns([5, 1])
            with col_title:
                is_active = conv["id"] == st.session_state.conversation_id
                label = f"{'🟢 ' if is_active else ''}{conv['title'][:30]}"
                if st.button(label, key=f"conv_{conv['id']}",
                             use_container_width=True, type="secondary"):
                    detail = fetch_conversation_detail(api_base, conv["id"])
                    if detail:
                        st.session_state.conversation_id = conv["id"]
                        st.session_state.messages = []
                        for m in detail["messages"]:
                            st.session_state.messages.append({
                                "role": m["role"],
                                "content": m["content"],
                                "question": m.get("question"),
                                "rewritten_query": m.get("rewritten_query"),
                                "sources": m.get("sources"),
                                "confidence": m.get("confidence"),
                            })
            with col_del:
                if st.button("🗑️", key=f"del_{conv['id']}", help="删除"):
                    if delete_conversation_api(api_base, conv["id"]):
                        st.session_state.conversation_list = fetch_conversations(api_base)
                        if st.session_state.conversation_id == conv["id"]:
                            st.session_state.messages = []
                            st.session_state.conversation_id = None

    # ── 我的收藏 ──
    if st.session_state.mysql_ok:
        st.divider()
        st.subheader("⭐ 我的收藏")

        # 缓存收藏列表，避免每次重跑都拉取
        if st.session_state.favorites_list is None:
            st.session_state.favorites_list = fetch_favorites(api_base)

        favs = st.session_state.favorites_list
        if not favs:
            st.caption("暂无收藏")
        for fav in favs[:10]:
            with st.container():
                if fav["fav_type"] == "conversation":
                    st.markdown(f"💬 对话收藏")
                elif fav["fav_type"] == "source":
                    data = fav.get("source_data")
                    if isinstance(data, str):
                        try:
                            data = json.loads(data)
                        except Exception:
                            data = {}
                    if data:
                        st.markdown(f"📌 {data.get('source', '')} {data.get('article_num', '')}")
                    else:
                        st.markdown("📌 条文收藏")
                if st.button("❌ 取消", key=f"unfav_{fav['id']}"):
                    delete_favorite(api_base, fav["id"])
                    st.session_state.favorites_list = None  # 下次重跑重新拉取

    # ── 文档管理 ──
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
                    time.sleep(1)
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
                    time.sleep(1)
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
                    time.sleep(1)
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

# 显示当前对话状态
if st.session_state.conversation_id:
    st.info(f"当前对话 ID: {st.session_state.conversation_id[:8]}...")

# 收藏整个对话按钮
if st.session_state.mysql_ok and st.session_state.conversation_id:
    if st.button("⭐ 收藏当前对话", key="fav_conv"):
        result = add_favorite(
            api_base, fav_type="conversation",
            conversation_id=st.session_state.conversation_id,
        )
        if result:
            st.toast("对话已收藏", icon="⭐")
            st.session_state.favorites_list = None  # 刷新收藏列表

# 历史消息回放
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("question"):
            st.caption(f"已改写查询：{msg['rewritten_query']}")
        if msg.get("sources"):
            render_sources(msg["sources"], msg_idx=idx,
                           conv_id=st.session_state.conversation_id)
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
        answer = ""
        rewritten = ""
        sources = []
        confidence = {}

        try:
            payload: dict = {
                "question": question,
                "use_reranker": use_reranker,
                "use_query_rewrite": use_query_rewrite,
            }
            if doc_type_filter:
                payload["doc_type_filter"] = doc_type_filter
            # 传入对话ID
            if st.session_state.conversation_id:
                payload["conversation_id"] = st.session_state.conversation_id

            response = requests.post(
                f"{api_base}/api/qa/stream",
                json=payload,
                timeout=180,
                stream=True,
            )
            response.raise_for_status()

            answer_box = st.empty()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or raw_line.startswith("data: [DONE]"):
                    continue

                line = raw_line
                if line.startswith("data: "):
                    line = line[6:]

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                if event.get("type") == "meta":
                    rewritten = event.get("rewritten_query", "") or ""
                    sources = event.get("sources", [])
                elif event.get("type") == "token":
                    answer += event.get("text", "")
                    answer_box.markdown(answer + "▌")
                elif event.get("type") == "done":
                    confidence = event.get("confidence", {})
                    break
                elif event.get("type") == "error":
                    answer = f"出错了: {event.get('message', '')}"
                    break

            answer_box.markdown(answer)
            if rewritten and rewritten != question:
                st.caption(f"已改写查询：{rewritten}")
            render_sources(sources, msg_idx=len(st.session_state.messages),
                           conv_id=st.session_state.conversation_id)
            if confidence:
                render_confidence(confidence)

        except RequestException as e:
            err_msg = _handle_api_error(e)
            st.error(err_msg)
            answer = err_msg

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "question": question,
            "rewritten_query": rewritten,
            "sources": sources,
            "confidence": confidence,
        })

        # 流式结束后，服务端已自动保存，但前端需要刷新对话列表
        st.session_state.conversation_list = fetch_conversations(api_base)
