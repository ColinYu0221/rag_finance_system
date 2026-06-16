"""
app.py
Streamlit 前端 — 金融制度 RAG 问答系统
纯 HTTP 客户端，所有后端调用走 FastAPI。
墨绿色 + 金色主题，顶部齿轮按钮展开横向设置面板。
"""

import json
import os
import base64
from pathlib import Path
from typing import Generator

import requests
import streamlit as st
from streamlit.components.v1 import html
from requests.exceptions import ConnectionError, RequestException, Timeout

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:9099")
# ===== 健康检查缓存 =====
_health_cache = {"ts": 0.0, "api_ok": False}

def cached_health_check(api_base: str) -> bool:
    now = _time.time()
    if now - _health_cache["ts"] < 30 and st.session_state.api_ok is not None:
        return _health_cache["api_ok"]
    ok = check_api_health(api_base)
    _health_cache.update(ts=now, api_ok=ok)
    return ok

# ===== 自定义 CSS - 墨绿+金色主题 =====
CUSTOM_CSS = """
<style>
    :root {
        --bg-base: #0c2318;
        --bg-surface: #143326;
        --bg-elevated: #1c4332;
        --bg-hover: #265540;
        --text-primary: #e8d589;
        --text-secondary: #bfae67;
        --text-tertiary: #8a7d4a;
        --border-default: #2a5a3a;
        --border-strong: #3d7a50;
        --accent: #d4a843;
        --accent-hover: #e8c460;
        --accent-muted: #3d3424;
        --success: #7ec99c;
        --warning: #e8c460;
        --error: #e06c6c;
        --radius: 8px;
    }

    .stApp { background-color: var(--bg-base); }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; }
    html, body, [data-testid="stAppViewContainer"] { background: var(--bg-base); color: var(--text-primary); }

    .main .block-container {
        padding: 1.5rem 1.5rem 6rem 1.5rem;
        max-width: 1100px;
        background: var(--bg-base);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    h1 {
        font-family: 'KaiTi', 'STKaiti', '楷体', 'AR PL UKai CN', serif;
        font-size: 36px;
        font-weight: 700;
        color: var(--text-primary);
        text-align: center;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
        text-shadow: 0 0 30px rgba(212, 168, 67, 0.15);
    }

    h2, h3, h4 { color: var(--text-primary); font-weight: 600; }
    h2 { font-size: 22px; margin-top: 1.5rem; }
    h3 { font-size: 17px; margin-top: 1.25rem; }
    h4 { font-size: 15px; }

    p, span, div, label { color: var(--text-primary); }
    .stCaption { color: var(--text-tertiary); font-size: 13px; text-align: center; }

    section[data-testid="stSidebar"] { display: none; }

    .dial-btn button {
        border-radius: 50% !important;
        width: 44px !important;
        height: 44px !important;
        min-width: 44px !important;
        min-height: 44px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 22px !important;
        background: var(--bg-elevated) !important;
        border: 2px solid var(--border-strong) !important;
        color: var(--accent) !important;
        transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        line-height: 1 !important;
    }

    .dial-btn button:hover {
        transform: rotate(270deg) scale(1.2) !important;
        border-color: var(--accent) !important;
        box-shadow: 0 0 20px rgba(212, 168, 67, 0.4) !important;
        background: var(--bg-hover) !important;
    }

    .dial-btn button:active {
        transform: rotate(270deg) scale(0.95) !important;
        transition: all 0.1s ease !important;
    }

    .stButton button {
        background: var(--accent);
        color: #1a1a1a;
        border: none;
        border-radius: var(--radius);
        padding: 0.55rem 1.2rem;
        font-size: 13.5px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s ease;
    }

    .stButton button:hover { background: var(--accent-hover); transform: translateY(-1px); }
    .stButton button:active { transform: translateY(0); }

    .stButton button[kind="secondary"] {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-strong);
    }

    .stButton button[kind="secondary"]:hover {
        background: var(--bg-hover);
        border-color: var(--text-tertiary);
    }

    .stButton button:disabled { background: var(--bg-hover); color: var(--text-tertiary); }

    .stTextInput input, .stTextArea textarea {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius);
        font-size: 14px;
        color: var(--text-primary);
        padding: 0.6rem 0.85rem;
        transition: all 0.12s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(212, 168, 67, 0.15);
        outline: none;
    }

    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: var(--text-tertiary); }
    .stTextInput label, .stTextArea label { font-size: 13px; font-weight: 500; color: var(--text-secondary); margin-bottom: 0.4rem; }

    .stSelectbox > div > div { background: var(--bg-elevated); border: 1px solid var(--border-default); border-radius: var(--radius); }
    .stSelectbox [data-baseweb="select"] { background: var(--bg-elevated); }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border-default);
        justify-content: center;
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        padding: 0 1.5rem;
        font-size: 14px;
        font-weight: 500;
        color: var(--text-secondary);
        border-bottom: 2px solid transparent;
        background: transparent;
        transition: all 0.15s ease;
    }

    .stTabs [data-baseweb="tab"]:hover { color: var(--text-primary); background: rgba(20, 51, 38, 0.5); }
    .stTabs [aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

    .stChatMessage { background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: var(--radius); padding: 1.25rem; margin-bottom: 0.75rem; }
    [data-testid="stChatMessage"] { background: var(--bg-surface); }

    .streamlit-expanderHeader {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius);
        font-size: 14px;
        font-weight: 500;
        color: var(--text-primary);
        padding: 0.7rem 1rem;
        transition: all 0.15s ease;
    }

    .streamlit-expanderHeader:hover { background: var(--bg-elevated); border-color: var(--border-strong); }
    .streamlit-expanderContent { background: var(--bg-surface); border: 1px solid var(--border-default); border-top: none; border-radius: 0 0 var(--radius) var(--radius); padding: 1rem; }

    [data-testid="stMetric"] { background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: var(--radius); padding: 1rem; }
    [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; font-size: 11px !important; font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.06em; }
    [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 26px !important; font-weight: 600 !important; margin-top: 0.2rem; }

    [data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius);
        padding: 1rem;
        background: var(--bg-surface);
    }

    .settings-panel {
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius);
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    .stSuccess, .stWarning, .stError, .stInfo { border-radius: var(--radius); border: 1px solid; padding: 0.8rem 1rem; font-size: 13.5px; font-weight: 500; }
    .stSuccess { background: #1a2e22; border-color: #3a6e4a; color: var(--success); }
    .stWarning { background: #2e2818; border-color: #6e5a2e; color: var(--warning); }
    .stError { background: #2e1a1a; border-color: #6e3a3a; color: var(--error); }
    .stInfo { background: var(--bg-surface); border-color: var(--border-default); color: var(--text-secondary); }

    .stProgress { background: var(--bg-elevated); border-radius: 4px; }
    .stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-hover)); }

    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border-strong);
        border-radius: var(--radius);
        padding: 1.5rem;
        background: var(--bg-surface);
        transition: all 0.15s ease;
    }

    [data-testid="stFileUploader"]:hover { border-color: var(--accent); background: var(--bg-elevated); }

    hr { border: none; border-top: 1px solid var(--border-default); margin: 1.5rem 0; }
    .stCheckbox label, .stRadio label, .stToggle label { font-size: 14px; color: var(--text-primary); }

    .qa-input-row {
        margin-top: 2rem;
        margin-bottom: 0.5rem;
        padding: 0.75rem 1rem;
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius);
    }
    .qa-input-row input {
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius);
        font-size: 14px;
        color: var(--text-primary);
        padding: 0.7rem 0.85rem;
    }
    .qa-input-row input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(212, 168, 67, 0.2);
    }
    .qa-input-row input::placeholder { color: var(--text-tertiary); }

    .st-emotion-cache-10trblm, .st-bd, .st-br, .st-bc, .st-emotion-cache-1xarl3l { color: var(--text-primary); }
    a { color: var(--accent); }
    .stDataFrame { background: var(--bg-surface); border: 1px solid var(--border-default); border-radius: var(--radius); }

    /* 收藏按钮样式 */
    .stButton > button[key*="fav_src"] {
        background: linear-gradient(135deg, #d4a843 0%, #b8922e 100%) !important;
        color: #1a1a1a !important;
        border: none !important;
        border-radius: 20px !important;
        padding: 0.3rem 0.8rem !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    .stButton > button[key*="fav_src"]:hover {
        background: linear-gradient(135deg, #e8c460 0%, #d4a843 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(212, 168, 67, 0.4) !important;
    }
    .stButton > button[key="fav_conversation"] {
        background: linear-gradient(135deg, #d4a843 0%, #b8922e 100%) !important;
        color: #1a1a1a !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
    }
    .stButton > button[key="fav_conversation"]:hover {
        background: linear-gradient(135deg, #e8c460 0%, #d4a843 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(212, 168, 67, 0.5) !important;
    }
</style>
"""

# ===== 页面配置 =====
st.set_page_config(
    page_title="金融制度知识问答系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 注入自定义 CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===== Session State 初始化 =====
if "api_base_url" not in st.session_state:
    st.session_state.api_base_url = DEFAULT_API_URL
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_ok" not in st.session_state:
    st.session_state.api_ok = None
if "last_health_check" not in st.session_state:
    st.session_state.last_health_check = 0
if "settings_open" not in st.session_state:
    st.session_state.settings_open = False
if "use_reranker" not in st.session_state:
    st.session_state.use_reranker = True
if "use_query_rewrite" not in st.session_state:
    st.session_state.use_query_rewrite = True
if "mode" not in st.session_state:
    st.session_state.mode = "全部"
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "conversation_list" not in st.session_state:
    st.session_state.conversation_list = []
if "favorites_list" not in st.session_state:
    st.session_state.favorites_list = None
if "jump_to_tab" not in st.session_state:
    st.session_state.jump_to_tab = None
if "jump_law_name" not in st.session_state:
    st.session_state.jump_law_name = None
if "jump_article_num" not in st.session_state:
    st.session_state.jump_article_num = None



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
    resp = requests.post(f"{api_base}/api/qa", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def ask_question_stream(
    api_base: str,
    question: str,
    doc_type_filter: str | None,
    use_reranker: bool,
    use_query_rewrite: bool,
    conversation_id: str | None = None,
) -> tuple[Generator[str, None, None], dict]:
    """流式问答：返回 (token_generator, metadata_holder)。"""
    payload: dict = {
        "question": question,
        "use_reranker": use_reranker,
        "use_query_rewrite": use_query_rewrite,
    }
    if doc_type_filter:
        payload["doc_type_filter"] = doc_type_filter
    if conversation_id:
        payload["conversation_id"] = conversation_id

    metadata: dict = {}

    def token_stream() -> Generator[str, None, None]:
        try:
            resp = requests.post(
                f"{api_base}/api/qa/stream",
                json=payload,
                timeout=180,
                stream=True,
            )
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "meta":
                        metadata["conversation_id"] = event.get("conversation_id")
                        metadata["sources"] = event.get("sources", [])
                        metadata["rewritten_query"] = event.get("rewritten_query")
                    elif event.get("type") == "token":
                        yield event.get("text", "")
                    elif event.get("type") == "done":
                        metadata["confidence"] = event.get("confidence", {})
                    elif event.get("type") == "error":
                        yield f"\n\n**{event.get('message', '未知错误')}**"
        except RequestException as e:
            yield f"\n\n**{_handle_api_error(e)}**"

    return token_stream(), metadata


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


# 分类管理 Helper

def fetch_categories(api_base: str) -> dict[str, list[str]]:
    resp = requests.get(f"{api_base}/api/categories", timeout=10)
    resp.raise_for_status()
    return resp.json().get("categories", {})


def rename_category_api(api_base: str, old_name: str, new_name: str) -> dict:
    resp = requests.put(
        f"{api_base}/api/categories/rename",
        json={"old_name": old_name, "new_name": new_name},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def delete_category_api(api_base: str, name: str) -> dict:
    resp = requests.delete(f"{api_base}/api/categories/{name}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_dictionary_items(api_base: str, item_type: str) -> list[dict]:
    resp = requests.get(f"{api_base}/api/dictionary/{item_type}", timeout=10)
    resp.raise_for_status()
    return resp.json().get("items", [])


def set_item_category_api(api_base: str, item_type: str, item_name: str, category: str) -> dict:
    resp = requests.put(
        f"{api_base}/api/dictionary/{item_name}/category",
        json={"item_type": item_type, "item_name": item_name, "category": category},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# 渲染 Helper

def render_sources(sources: list[dict], conv_id: str | None = None):
    if not sources:
        return
    st.markdown(f"**溯源条文 ({len(sources)} 条)**")
    for i, src in enumerate(sources, 1):
        conf_color = "green" if src["score"] > 0.7 else "orange" if src["score"] > 0.4 else "red"
        st.markdown(
            f"**[{i}] [{src['source']} {src['article_num']}]** "
            f":{conf_color}[{src['score']:.2%}]"
        )
        st.text(src["text"][:300] + "..." if len(src["text"]) > 300 else src["text"])
        if st.session_state.api_ok:
            if st.button("⭐ 收藏此条文", key=f"fav_src_{i}", help="收藏此条文"):
                result = add_favorite_api(
                    st.session_state.api_base_url, "source", conv_id, src
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


# 条文关联查询 Helper

@st.cache_data(ttl=300)
def get_law_names(api_base: str) -> list[str]:
    try:
        resp = requests.get(f"{api_base}/api/laws", timeout=10)
        resp.raise_for_status()
        return resp.json().get("law_names", [])
    except (ConnectionError, Timeout, RequestException):
        return []


def query_article_relations(api_base: str, law_name: str, article_num: str) -> dict:
    resp = requests.post(
        f"{api_base}/api/articles/relations",
        json={"law_name": law_name, "article_num": article_num},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# 流程图 Helper

def flowchart_from_image_api(api_base: str, image_base64: str, prefer_multimodal: bool = True) -> dict:
    resp = requests.post(
        f"{api_base}/api/flowchart/image",
        json={"image_base64": image_base64, "prefer_multimodal": prefer_multimodal},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def flowchart_from_text_api(api_base: str, text: str) -> dict:
    resp = requests.post(
        f"{api_base}/api/flowchart/text",
        json={"text": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def render_mermaid(mermaid_str: str, height: int = 600):
    escaped = mermaid_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0; padding: 20px;
            background: #0c2318;
            font-family: sans-serif;
        }}
        .mermaid {{
            font-family: sans-serif;
            background: #143326;
            border: 1px solid #2a5a3a;
            border-radius: 8px;
            padding: 24px;
        }}
    </style>
</head>
<body>
    <div class="mermaid">
{escaped}
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'dark',
            themeVariables: {{
                primaryColor: '#1c4332',
                primaryTextColor: '#e8d589',
                primaryBorderColor: '#3d7a50',
                lineColor: '#d4a843',
                secondaryColor: '#143326',
                tertiaryColor: '#265540',
                fontSize: '14px'
            }},
            flowchart: {{ useMaxWidth: true, htmlLabels: true, curve: 'basis', padding: 15 }}
        }});
    </script>
</body>
</html>"""
    html(html_content, height=height)


# ===== 对话历史 & 收藏 Helper =====

def _safe_parse(val):
    """安全解析 JSON 字段：已经是 dict/list 则直接返回，字符串则 json.loads。"""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None
    return val

def fetch_conversations(api_base: str) -> list[dict]:
    try:
        resp = requests.get(f"{api_base}/api/conversations", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (ConnectionError, Timeout, RequestException):
        return []


def fetch_conversation_detail(api_base: str, conv_id: str) -> dict | None:
    try:
        resp = requests.get(f"{api_base}/api/conversations/{conv_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (ConnectionError, Timeout, RequestException):
        return None


def delete_conversation_api(api_base: str, conv_id: str) -> bool:
    try:
        resp = requests.delete(f"{api_base}/api/conversations/{conv_id}", timeout=10)
        resp.raise_for_status()
        return True
    except (ConnectionError, Timeout, RequestException):
        return False


def fetch_favorites(api_base: str) -> list[dict]:
    try:
        resp = requests.get(f"{api_base}/api/favorites", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (ConnectionError, Timeout, RequestException):
        return []


def add_favorite_api(api_base: str, fav_type: str, conversation_id: str | None = None,
                     source_data: dict | None = None) -> dict | None:
    payload: dict = {"fav_type": fav_type}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    if source_data:
        payload["source_data"] = source_data
    try:
        resp = requests.post(f"{api_base}/api/favorites", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (ConnectionError, Timeout, RequestException):
        return None


def delete_favorite(api_base: str, fav_id: str) -> bool:
    try:
        resp = requests.delete(f"{api_base}/api/favorites/{fav_id}", timeout=10)
        resp.raise_for_status()
        return True
    except (ConnectionError, Timeout, RequestException):
        return False

# ====================================================================

# 页面加载时立即执行健康检查
import time as _time_init
_now_init = _time_init.time()
if st.session_state.api_ok is None or (_now_init - st.session_state.last_health_check) > 10:
    st.session_state.api_ok = check_api_health(st.session_state.api_base_url)
    st.session_state.last_health_check = _now_init

api_base = st.session_state.api_base_url

# API 在线时预加载对话历史和收藏（避免首次加载需要手动刷新）
if st.session_state.api_ok:
    if not st.session_state.conversation_list:
        try:
            st.session_state.conversation_list = fetch_conversations(api_base)
        except Exception:
            st.session_state.conversation_list = []
    if st.session_state.favorites_list is None:
        try:
            st.session_state.favorites_list = fetch_favorites(api_base)
        except Exception:
            st.session_state.favorites_list = []

# ---- 顶部栏：标题 + 齿轮 ----
col_title, col_gear = st.columns([0.92, 0.08])
with col_title:
    st.markdown("# 金融制度知识问答系统")
    st.caption("基于 RAG 的金融法规智能问答  bge-small-zh-v1.5 + Qwen2.5")
with col_gear:
    st.markdown('<div class="dial-btn">', unsafe_allow_html=True)
    if st.button("", icon="⚙", key="toggle_settings", type="secondary", help="设置"):
        st.session_state.settings_open = not st.session_state.settings_open
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ---- 标签页 ----
tab_qa, tab_relations, tab_categories, tab_flowchart, tab_favorites = st.tabs(
    ["智能问答", "条文关联查询", "标签分类管理", "流程图生成", "我的收藏"]
)

# JS 编程式切换标签页（在收藏中点击跳转时使用）
_switch_tab = st.session_state.pop("jump_to_tab", None)
if _switch_tab is not None:
    st.html(f"""<script>
setTimeout(function(){{
    var el = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
    if (el[{_switch_tab}]) el[{_switch_tab}].click();
}}, 50);
</script>""")

# Tab 1: 智能问答（左侧对话列表 + 右侧聊天区域）
with tab_qa:
    _mode_map = {"全部": None, "仅法条": "law", "仅案例": "case", "仅其他": "other"}
    doc_type_filter = _mode_map.get(st.session_state.mode)

    col_sidebar, col_chat = st.columns([1, 4])

    with col_sidebar:
        st.markdown("#### 对话历史")
        if st.button("📝 新对话", type="primary", use_container_width=True, key="new_conv"):
            st.session_state.messages = []
            st.session_state.conversation_id = None

        if st.session_state.api_ok:
            if not st.session_state.conversation_list:
                st.session_state.conversation_list = fetch_conversations(api_base)

            for conv in st.session_state.conversation_list:
                is_active = conv["id"] == st.session_state.conversation_id
                label = f"{'🟢 ' if is_active else ''}{conv['title'][:25]}"
                c1, c2 = st.columns([4, 1])
                with c1:
                    if st.button(label, key=f"conv_{conv['id']}", use_container_width=True,
                                 type="secondary" if not is_active else "primary"):
                        detail = fetch_conversation_detail(api_base, conv["id"])
                        if detail:
                            st.session_state.conversation_id = conv["id"]
                            st.session_state.messages = []
                            for m in detail.get("messages", []):
                                st.session_state.messages.append({
                                    "role": m["role"],
                                    "content": m["content"],
                                    "question": m.get("question"),
                                    "rewritten_query": m.get("rewritten_query"),
                                    "sources": _safe_parse(m.get("sources")),
                                    "confidence": _safe_parse(m.get("confidence")),
                                })
                            st.rerun()
                with c2:
                    if st.button("🗑️", key=f"del_{conv['id']}", help="删除"):
                        if delete_conversation_api(api_base, conv["id"]):
                            st.session_state.conversation_list = fetch_conversations(api_base)
                            if st.session_state.conversation_id == conv["id"]:
                                st.session_state.messages = []
                                st.session_state.conversation_id = None

            if not st.session_state.conversation_list:
                st.caption("暂无对话记录")

    with col_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("question"):
                    st.caption(f"已改写查询: {msg['rewritten_query']}")
                if msg.get("sources"):
                    render_sources(msg["sources"], st.session_state.conversation_id)
                if msg.get("confidence"):
                    render_confidence(msg["confidence"])

        with st.form(key="qa_form", clear_on_submit=True, border=False):
            col_input, col_btn = st.columns([8, 1])
            with col_input:
                question = st.text_input(
                    "输入问题", placeholder="请输入您的金融法规问题...",
                    key="qa_input", label_visibility="collapsed",
                    disabled=not st.session_state.api_ok,
                )
            with col_btn:
                submitted = st.form_submit_button("发送", type="primary", disabled=not st.session_state.api_ok, use_container_width=True)

        if submitted and question and question.strip():
            _q = question.strip()
            with st.chat_message("user"):
                st.markdown(_q)
            st.session_state.messages.append({"role": "user", "content": _q})

            with st.chat_message("assistant"):
                token_gen, metadata = ask_question_stream(
                    api_base, _q, doc_type_filter,
                    st.session_state.use_reranker, st.session_state.use_query_rewrite,
                    conversation_id=st.session_state.conversation_id,
                )
                answer = st.write_stream(token_gen)

                if answer is None or answer.strip() == "":
                    answer = "（未获取到回答）"

                rewritten = metadata.get("rewritten_query")
                sources = metadata.get("sources", [])
                confidence = metadata.get("confidence", {})
                new_conv_id = metadata.get("conversation_id")

                if new_conv_id and not st.session_state.conversation_id:
                    st.session_state.conversation_id = new_conv_id
                    st.session_state.conversation_list = fetch_conversations(api_base)

                if rewritten and rewritten != _q and rewritten != answer:
                    st.caption(f"已改写查询: {rewritten}")
                if confidence:
                    render_confidence(confidence)
                render_sources(sources, st.session_state.conversation_id)

                if st.session_state.api_ok and st.session_state.conversation_id:
                    st.divider()
                    col_fav1, col_fav2 = st.columns(2)
                    with col_fav1:
                        if st.button("⭐ 收藏对话", key="fav_conversation", 
                                   help="收藏此对话", use_container_width=True):
                            result = add_favorite_api(
                                api_base, "conversation", st.session_state.conversation_id
                            )
                            if result:
                                st.toast("对话已收藏", icon="⭐")
                                st.session_state.favorites_list = None
                    with col_fav2:
                        if st.button("📋 复制回答", key="copy_answer", 
                                   help="复制回答到剪贴板", use_container_width=True):
                            st.toast("已复制到剪贴板", icon="📋")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "question": _q,
                    "rewritten_query": rewritten,
                    "sources": sources,
                    "confidence": confidence,
                })

                st.session_state.conversation_list = fetch_conversations(api_base)

# Tab 2: 条文关联查询
with tab_relations:
    st.markdown("### 条文关联查询")
    st.caption("查询法规条文之间的引用关系、所属文档及关联法规")

    # 从收藏跳转时预填条文
    _jump_law = st.session_state.pop("jump_law_name", None)
    _jump_article = st.session_state.pop("jump_article_num", None)
    if _jump_law:
        st.session_state["article_rel_law_name"] = _jump_law
    if _jump_article:
        st.session_state["article_rel_num"] = _jump_article

    law_names = get_law_names(api_base)
    law_names_sorted = sorted(law_names) if law_names else []

    col1, col2 = st.columns([3, 1])
    with col1:
        if law_names_sorted:
            selected_law = st.selectbox(
                "选择法规", options=law_names_sorted, index=None,
                placeholder="输入或选择法规名称...", key="article_rel_law_name",
            )
        else:
            selected_law = st.text_input(
                "法规名称", placeholder="例如: 中华人民共和国公司法",
                key="article_rel_law_name", disabled=not st.session_state.api_ok,
            )
    with col2:
        article_num = st.text_input(
            "条文编号", placeholder="例如: 16",
            key="article_rel_num", disabled=not st.session_state.api_ok,
        )

    if st.button("查询关联关系", type="primary",
                 disabled=not st.session_state.api_ok or not selected_law or not article_num):
        with st.spinner("正在查询知识图谱..."):
            try:
                data = query_article_relations(api_base, selected_law, article_num)
            except RequestException as e:
                st.error(_handle_api_error(e))
                data = None

        if data:
            target = data.get("target")
            if target:
                st.markdown("### 目标条文")
                with st.container(border=True):
                    st.markdown(f"**{target['law_name']}**  第{target['article_num']}条")
                    st.text(target["text"])

                incoming = data.get("incoming_refs", [])
                st.markdown(f"### 引用此条文的条文 ({len(incoming)})")
                if incoming:
                    for i, ref in enumerate(incoming, 1):
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {ref['law_name']} 第{ref['article_num']}条**")
                            st.caption(f"来源: {ref['source']}")
                            st.text(ref["text"][:300] + ("..." if len(ref["text"]) > 300 else ""))
                else:
                    st.info("暂无其他条文引用此条文")

                outgoing = data.get("outgoing_refs", [])
                st.markdown(f"### 此条文引用的条文 ({len(outgoing)})")
                if outgoing:
                    for i, ref in enumerate(outgoing, 1):
                        target_label = ""
                        if ref.get("target_law"):
                            target_label = f" -> {ref['target_law']} 第{ref.get('target_article', '')}条"
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {ref['law_name']} 第{ref['article_num']}条**{target_label}")
                            st.caption(f"来源: {ref['source']}")
                            st.text(ref["text"][:300] + ("..." if len(ref["text"]) > 300 else ""))
                else:
                    st.info("此条文未引用其他条文")

                parent = data.get("parent_document")
                if parent:
                    st.markdown("### 所属文档")
                    with st.container(border=True):
                        st.markdown(f"**{parent['name']}**")
                        st.caption(f"类型: {parent.get('doc_type', '')} | 来源: {parent.get('source', '')}")

                related_docs = data.get("related_documents", [])
                st.markdown(f"### 关联法规 ({len(related_docs)})")
                if related_docs:
                    for i, rd in enumerate(related_docs, 1):
                        direction_icon = "->" if rd["direction"] == "outgoing" else "<-"
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {direction_icon} {rd['name']}**")
                            st.caption(f"关系类型: {rd['relation_type']} | 方向: {rd['direction']}")
                else:
                    st.info("无关联法规")

                related_articles = data.get("related_articles", [])
                st.markdown(f"### 关联法规示例条文 ({len(related_articles)})")
                if related_articles:
                    for i, ra in enumerate(related_articles, 1):
                        with st.container(border=True):
                            st.markdown(f"**[{i}] {ra['law_name']} 第{ra['article_num']}条**")
                            st.caption(f"来源: {ra['source']}")
                            st.text(ra["text"][:300] + ("..." if len(ra["text"]) > 300 else ""))
                else:
                    st.info("关联法规中暂无示例条文")
            else:
                _a = article_num.strip()
                if _a.startswith("第") and _a.endswith("条"):
                    _a = _a[1:-1]
                st.warning(f"未找到 '{selected_law}' 第{_a}条")
    else:
        st.info('请输入法规名称和条文编号，点击"查询关联关系"查看引用关系')
        if law_names_sorted:
            st.caption(f"知识图谱中现有 {len(law_names_sorted)} 部法规可供查询")

# Tab 3: 标签分类管理
with tab_categories:
    st.markdown("### 标签分类管理")
    st.caption("管理金融词典中术语、法规和机构的分类标签")

    api_ready = st.session_state.api_ok

    if not api_ready:
        st.warning("API 服务未连接，请先启动 FastAPI 后端")
    else:
        try:
            categories = fetch_categories(api_base)
        except RequestException as e:
            st.error(_handle_api_error(e))
            categories = {}

        all_categories: list[str] = []
        for cats in categories.values():
            for c in cats:
                if c not in all_categories:
                    all_categories.append(c)
        all_categories.sort()

        sub_cat, sub_item = st.tabs(["分类概览", "条目管理"])

        with sub_cat:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 术语分类")
                term_cats = categories.get("term", [])
                if term_cats:
                    for c in term_cats:
                        st.text(c)
                else:
                    st.caption("（无）")

            with col_b:
                st.markdown("#### 法规分类 / 机构分类")
                law_cats = categories.get("law", [])
                auth_cats = categories.get("authority", [])
                for c in law_cats + auth_cats:
                    st.text(c)
                if not law_cats and not auth_cats:
                    st.caption("（无）")

            st.divider()

            st.markdown("#### 重命名分类")
            if all_categories:
                col_old, col_new, col_btn = st.columns([2, 2, 1])
                with col_old:
                    rename_old = st.selectbox("原分类名", options=all_categories, key="rename_old")
                with col_new:
                    rename_new = st.text_input("新分类名", key="rename_new", placeholder="输入新名称...")
                with col_btn:
                    st.write("")
                    if st.button("重命名", key="btn_rename", disabled=not rename_new.strip()):
                        try:
                            result = rename_category_api(api_base, rename_old, rename_new.strip())
                            st.success(
                                f"已重命名: {rename_old} -> {rename_new.strip()} "
                                f"(术语{result['affected']['term']}, 法规{result['affected']['law']}, 机构{result['affected']['authority']})"
                            )
                            st.rerun()
                        except RequestException as e:
                            st.error(_handle_api_error(e))
            else:
                st.info("暂无分类可重命名")

            st.markdown("#### 删除分类")
            st.caption("删除后该分类下条目的 category 将被置空")
            if all_categories:
                col_del, col_btn2 = st.columns([3, 1])
                with col_del:
                    delete_target = st.selectbox("选择要删除的分类", options=all_categories, key="delete_cat")
                with col_btn2:
                    st.write("")
                    if st.button("删除", key="btn_delete", type="secondary"):
                        try:
                            result = delete_category_api(api_base, delete_target)
                            st.success(
                                f"已删除分类 '{delete_target}' "
                                f"(术语{result['affected']['term']}, 法规{result['affected']['law']}, 机构{result['affected']['authority']})"
                            )
                            st.rerun()
                        except RequestException as e:
                            st.error(_handle_api_error(e))
            else:
                st.info("暂无分类可删除")

        with sub_item:
            item_type_label = st.radio(
                "条目类型",
                options=["术语 (term)", "法规 (law)", "机构 (authority)"],
                horizontal=True,
                key="item_type_radio",
            )
            item_type_map = {"术语 (term)": "term", "法规 (law)": "law", "机构 (authority)": "authority"}
            selected_item_type = item_type_map[item_type_label]

            try:
                items = fetch_dictionary_items(api_base, selected_item_type)
            except RequestException as e:
                st.error(_handle_api_error(e))
                items = []

            if items:
                st.caption(f"共 {len(items)} 条，点击修改分类")
                items_by_cat: dict[str, list[dict]] = {}
                for it in items:
                    c = it.get("category", "") or "（未分类）"
                    if c not in items_by_cat:
                        items_by_cat[c] = []
                    items_by_cat[c].append(it)

                for cat_name, cat_items in items_by_cat.items():
                    with st.expander(f"{cat_name} ({len(cat_items)} 条)"):
                        for it in cat_items:
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.text(it["name"])
                            with c2:
                                new_cat = st.selectbox(
                                    "分类",
                                    options=["（未分类）"] + all_categories,
                                    index=(
                                        all_categories.index(it.get("category", "")) + 1
                                        if it.get("category", "") in all_categories
                                        else 0
                                    ),
                                    key=f"cat_{selected_item_type}_{it['name']}",
                                    label_visibility="collapsed",
                                )
                                target_cat = "" if new_cat == "（未分类）" else new_cat
                                if target_cat != (it.get("category", "") or ""):
                                    try:
                                        set_item_category_api(
                                            api_base, selected_item_type, it["name"], target_cat
                                        )
                                        st.rerun()
                                    except RequestException as e:
                                        st.error(_handle_api_error(e))
            else:
                st.info(f"暂无 {selected_item_type} 类型的条目")

# Tab 4: 流程图生成
with tab_flowchart:
    st.markdown("### 流程图生成")
    st.caption("从图片或文本中识别流程步骤，自动生成 Mermaid 流程图")

    if "flowchart_mermaid" not in st.session_state:
        st.session_state.flowchart_mermaid = ""
    if "flowchart_source" not in st.session_state:
        st.session_state.flowchart_source = ""
    if "flowchart_error" not in st.session_state:
        st.session_state.flowchart_error = None

    input_mode = st.radio(
        "输入方式", options=["图片上传", "文本输入"], horizontal=True, key="flowchart_input_mode",
    )

    if input_mode == "图片上传":
        flowchart_image = st.file_uploader(
            "上传流程图片", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
            key="flowchart_image_uploader", help="上传包含流程/步骤描述的图片",
        )

        if flowchart_image:
            st.image(flowchart_image, caption="已上传图片", use_container_width=True)

        path_mode = st.radio(
            "生成路径",
            options=["自动（优先多模态）", "仅 OCR + 文本 LLM", "仅多模态"],
            horizontal=True, key="flowchart_path_mode",
        )
        prefer_multimodal_map = {
            "自动（优先多模态）": True,
            "仅 OCR + 文本 LLM": False,
            "仅多模态": True,
        }
        prefer_multimodal = prefer_multimodal_map[path_mode]

        if st.button("生成流程图", type="primary",
                     disabled=not st.session_state.api_ok or flowchart_image is None,
                     key="flowchart_generate_btn") and flowchart_image:
            img_bytes = flowchart_image.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
            with st.spinner("正在生成流程图..."):
                try:
                    result = flowchart_from_image_api(api_base, img_base64, prefer_multimodal)
                    st.session_state.flowchart_mermaid = result.get("mermaid", "")
                    st.session_state.flowchart_source = result.get("source", "")
                    st.session_state.flowchart_error = result.get("error")
                except RequestException as e:
                    st.session_state.flowchart_mermaid = ""
                    st.session_state.flowchart_error = _handle_api_error(e)
    else:
        flowchart_text = st.text_area(
            "输入法规文本", placeholder="粘贴包含流程/步骤描述的法规文本...",
            height=200, key="flowchart_text_input",
        )
        if st.button("生成流程图", type="primary",
                     disabled=not st.session_state.api_ok or not flowchart_text.strip(),
                     key="flowchart_generate_btn_text") and flowchart_text.strip():
            with st.spinner("正在从文本生成流程图..."):
                try:
                    result = flowchart_from_text_api(api_base, flowchart_text.strip())
                    st.session_state.flowchart_mermaid = result.get("mermaid", "")
                    st.session_state.flowchart_source = result.get("source", "")
                    st.session_state.flowchart_error = result.get("error")
                except RequestException as e:
                    st.session_state.flowchart_mermaid = ""
                    st.session_state.flowchart_error = _handle_api_error(e)

    mermaid_str = st.session_state.flowchart_mermaid
    source = st.session_state.flowchart_source
    error = st.session_state.flowchart_error

    if error:
        st.warning(f"生成提示: {error}")

    if mermaid_str:
        source_label = {"multimodal": "多模态 LLM", "ocr+llm": "OCR + 文本 LLM", "text": "文本 LLM"}.get(source, source)
        st.caption(f"生成路径: {source_label}")

        st.markdown("### 流程图")
        render_mermaid(mermaid_str, height=600)

        st.markdown("### Mermaid 源码（可编辑后重新渲染）")
        edited_mermaid = st.text_area(
            "Mermaid 源码", value=mermaid_str, height=300, key="flowchart_mermaid_editor",
        )
        if edited_mermaid != mermaid_str and edited_mermaid.strip():
            st.session_state.flowchart_mermaid = edited_mermaid.strip()
            st.markdown("### 修改后流程图")
            render_mermaid(edited_mermaid.strip(), height=600)

        st.download_button(
            "下载 Mermaid 源码", data=mermaid_str,
            file_name="flowchart.mmd", mime="text/plain", key="flowchart_download_btn",
        )
    elif not error:
        st.info("请上传图片或输入文本，点击 生成流程图 开始")

# Tab 5: 我的收藏
with tab_favorites:
    st.markdown("### 我的收藏")
    st.caption("管理收藏的对话和条文")

    if not st.session_state.api_ok:
        st.warning("API 未连接，无法加载收藏")
    else:
        if st.session_state.favorites_list is None:
            st.session_state.favorites_list = fetch_favorites(api_base)
        favs = st.session_state.favorites_list

        if not favs:
            st.info("暂无收藏内容。在智能问答中点击 ⭐ 收藏对话或条文。")
        else:
            col_type, col_refresh = st.columns([3, 1])
            with col_type:
                fav_filter = st.selectbox(
                    "筛选类型", ["全部", "对话", "条文"], key="fav_filter_select"
                )
            with col_refresh:
                if st.button("🔄 刷新", key="refresh_favs"):
                    st.session_state.favorites_list = None
                    st.rerun()

            filter_map = {"全部": None, "对话": "conversation", "条文": "source"}
            filtered_type = filter_map.get(fav_filter)

            display_favs = [f for f in favs if filtered_type is None or f["fav_type"] == filtered_type]

            if not display_favs:
                st.info(f"没有{fav_filter}类型的收藏")
            else:
                for fav in display_favs:
                    with st.container():
                        c1, c2 = st.columns([5, 1])
                        with c1:
                            if fav["fav_type"] == "conversation":
                                conv_title = fav.get("conversation_id", "")[:12] or "对话"
                                st.markdown(f"💬 **对话** - `{conv_title}...`")
                                st.caption(f"收藏时间: {fav.get('created_at', '未知')}")
                                if st.button("查看对话", key=f"view_conv_{fav['id']}", use_container_width=True):
                                    detail = fetch_conversation_detail(api_base, fav["conversation_id"])
                                    if detail:
                                        st.session_state.conversation_id = fav["conversation_id"]
                                        st.session_state.messages = []
                                        for m in detail.get("messages", []):
                                            st.session_state.messages.append({
                                                "role": m["role"],
                                                "content": m["content"],
                                                "question": m.get("question"),
                                                "rewritten_query": m.get("rewritten_query"),
                                                "sources": _safe_parse(m.get("sources")),
                                                "confidence": _safe_parse(m.get("confidence")),
                                            })
                                        st.session_state.jump_to_tab = 0
                                        st.rerun()
                            elif fav["fav_type"] == "source":
                                data = fav.get("source_data")
                                if isinstance(data, str):
                                    try:
                                        data = json.loads(data)
                                    except Exception:
                                        data = {}
                                source_name = data.get("source", "未知") if data else "未知"
                                article_num = data.get("article_num", "") if data else ""
                                st.markdown(f"📌 **条文** - {source_name} {article_num}")
                                st.caption(f"收藏时间: {fav.get('created_at', '未知')}")
                                if data:
                                    st.text(data.get("text", "")[:150] + "..." if len(data.get("text", "")) > 150 else data.get("text", ""))
                                if st.button("查看条文", key=f"view_src_{fav['id']}", use_container_width=True):
                                    st.session_state.jump_law_name = data.get("source", "") if data else ""
                                    st.session_state.jump_article_num = data.get("article_num", "") if data else ""
                                    st.session_state.jump_to_tab = 1
                                    st.rerun()
                        with c2:
                            if st.button("❌", key=f"del_fav_{fav['id']}", help="取消收藏"):
                                delete_favorite(api_base, fav["id"])
                                st.session_state.favorites_list = None
                                st.rerun()
                        st.divider()

# ---- 设置面板（点击齿轮后展开在页面底部） ----
if st.session_state.settings_open:
    st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
    with st.container():
        st.markdown("### 系统设置")

        # 第1行：API 配置
        c1, c2 = st.columns([3, 1])
        with c1:
            url_input = st.text_input("API 服务器地址", value=st.session_state.api_base_url, key="api_url_input")
            if url_input.strip() != st.session_state.api_base_url:
                st.session_state.api_base_url = url_input.strip()
                st.session_state.api_ok = None
        with c2:
            import time as _time
            _now = _time.time()
            if st.session_state.api_ok is None or (_now - st.session_state.last_health_check) > 10:
                st.session_state.api_ok = check_api_health(st.session_state.api_base_url)
                st.session_state.last_health_check = _now
            if st.session_state.api_ok:
                st.success("API 在线")
            else:
                st.warning("API 离线")

        st.divider()

        # 第2行：检索配置（3个水平排列）
        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state.use_reranker = st.toggle(
                "启用 Reranker 精排", value=st.session_state.use_reranker, key="toggle_reranker"
            )
        with c2:
            st.session_state.use_query_rewrite = st.toggle(
                "启用查询重写", value=st.session_state.use_query_rewrite, key="toggle_rewrite"
            )
        with c3:
            _mode_index = ["全部", "仅法条", "仅案例", "仅其他"].index(st.session_state.mode) if st.session_state.mode in ["全部", "仅法条", "仅案例", "仅其他"] else 0
            st.session_state.mode = st.selectbox("检索范围", ["全部", "仅法条", "仅案例", "仅其他"], index=_mode_index, key="select_mode")

        st.divider()

        # 第3行：文档上传
        st.markdown("#### 文档管理")
        c1, c2 = st.columns([1, 2])
        with c1:
            doc_type_label = st.selectbox("文档类型", ["法规 (law)", "案例 (case)", "其他 (other)"], index=0, key="doc_type_select")
            doc_type_map = {"法规 (law)": "law", "案例 (case)": "case", "其他 (other)": "other"}
            doc_type = doc_type_map[doc_type_label]
        with c2:
            uploaded_file = st.file_uploader(
                "上传文档",
                type=["pdf", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
                help="支持 PDF、TXT 及图片",
                key="settings_uploader",
            )

        gen_flowchart = False
        if uploaded_file and Path(uploaded_file.name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}:
            st.info("图片将通过 OCR 识别后索引")
            gen_flowchart = st.checkbox("同时生成流程图", value=False, key="gen_flowchart_cb")

        if uploaded_file:
            if st.button("解析并建立索引", type="primary", disabled=not st.session_state.api_ok, key="btn_index"):
                try:
                    with st.spinner("上传中..."):
                        upload_resp = upload_file(st.session_state.api_base_url, uploaded_file.getvalue(), uploaded_file.name, doc_type)
                    file_path = upload_resp["file_path"]
                    with st.spinner("索引中..."):
                        index_resp = index_file(st.session_state.api_base_url, file_path, doc_type)
                    st.success(f"索引完成 {index_resp['chunk_count']} 个片段")

                    is_image = Path(uploaded_file.name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
                    if is_image and gen_flowchart:
                        with st.spinner("生成流程图中..."):
                            img_base64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
                            flow_result = flowchart_from_image_api(st.session_state.api_base_url, img_base64, prefer_multimodal=True)
                        if flow_result.get("success"):
                            st.markdown("#### 流程图")
                            render_mermaid(flow_result["mermaid"], height=500)
                            with st.expander("查看 Mermaid 源码"):
                                st.code(flow_result["mermaid"], language="mermaid")
                        else:
                            st.info(f"生成提示: {flow_result.get('error', '未能识别流程')}")
                except RequestException as e:
                    st.error(_handle_api_error(e))

        st.divider()

        # 第3.5行：我的收藏（点击可跳转）
        st.markdown("#### 我的收藏")
        if st.session_state.api_ok:
            if st.session_state.favorites_list is None:
                st.session_state.favorites_list = fetch_favorites(api_base)
            favs = st.session_state.favorites_list
            if not favs:
                st.caption("暂无收藏")
            for fav in favs[:10]:
                c1, c2 = st.columns([4, 1])
                with c1:
                    if fav["fav_type"] == "conversation":
                        conv_title = fav.get("conversation_id", "")[:8] or "对话"
                        if st.button(f"💬 {conv_title}...", key=f"jump_conv_{fav['id']}",
                                     use_container_width=True, help="跳转到此对话"):
                            detail = fetch_conversation_detail(api_base, fav["conversation_id"])
                            if detail:
                                st.session_state.conversation_id = fav["conversation_id"]
                                st.session_state.messages = []
                                for m in detail.get("messages", []):
                                    st.session_state.messages.append({
                                        "role": m["role"],
                                        "content": m["content"],
                                        "question": m.get("question"),
                                        "rewritten_query": m.get("rewritten_query"),
                                        "sources": _safe_parse(m.get("sources")),
                                        "confidence": _safe_parse(m.get("confidence")),
                                    })
                                st.session_state.jump_to_tab = 0
                                st.session_state.favorites_list = None
                                st.rerun()
                    elif fav["fav_type"] == "source":
                        data = fav.get("source_data")
                        if isinstance(data, str):
                            try:
                                data = json.loads(data)
                            except Exception:
                                data = {}
                        label = f"📌 {data.get('source', '')} {data.get('article_num', '')}" if data else "📌 条文收藏"
                        if st.button(label, key=f"jump_src_{fav['id']}",
                                     use_container_width=True, help="跳转到此条文"):
                            st.session_state.jump_law_name = data.get("source", "")
                            st.session_state.jump_article_num = data.get("article_num", "")
                            st.session_state.jump_to_tab = 1
                            st.session_state.favorites_list = None
                            st.rerun()
                with c2:
                    if st.button("❌", key=f"unfav_{fav['id']}", help="取消收藏"):
                        delete_favorite(api_base, fav["id"])
                        st.session_state.favorites_list = None
                        st.rerun()
        else:
            st.caption("API 未连接")

        st.divider()

        # 第4行：批量导入
        st.markdown("#### 批量导入")
        if st.button("一键导入 txt_files", type="secondary", key="batch_import",
                     disabled=not st.session_state.api_ok):
            import glob as _glob
            from pathlib import Path as _Path
            _txt_dir = _Path(__file__).resolve().parent / "src" / "txt_files"
            _txt_files = sorted(_glob.glob(str(_txt_dir / "*.txt")))
            if not _txt_files:
                st.warning(f"未找到 .txt 文件: {_txt_dir}")
            else:
                progress = st.progress(0, text=f"0/{len(_txt_files)}")
                total_chunks = 0
                errors = 0
                for i, fp in enumerate(_txt_files):
                    fname = _Path(fp).name
                    try:
                        with open(fp, "rb") as fh:
                            up = upload_file(st.session_state.api_base_url, fh.read(), fname, "law")
                        ix = index_file(st.session_state.api_base_url, up["file_path"], "law")
                        total_chunks += ix["chunk_count"]
                    except Exception as e:
                        st.warning(f"跳过 {fname}: {e}")
                        errors += 1
                    progress.progress((i + 1) / len(_txt_files), text=f"{i + 1}/{len(_txt_files)}  ({total_chunks} chunks)")
                st.success(f"完成 {len(_txt_files) - errors} 个文件 {total_chunks} 个片段")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # 设置关闭时保持默认值
    st.session_state.use_reranker = st.session_state.get("use_reranker", True)
    st.session_state.use_query_rewrite = st.session_state.get("use_query_rewrite", True)
    st.session_state.mode = st.session_state.get("mode", "全部")