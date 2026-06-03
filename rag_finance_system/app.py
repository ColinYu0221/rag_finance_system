"""
app.py
Streamlit前端 - 金融制度RAG问答系统
功能：文档上传、问答交互、溯源展示、可信度评分
"""

import sys
import os
import time
from pathlib import Path

import streamlit as st
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

# ===== 页面配置 =====
st.set_page_config(
    page_title="金融制度知识问答系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ===== 缓存：全局组件只初始化一次 =====
@st.cache_resource(show_spinner="加载模型中，请稍候...")
def load_components(use_reranker: bool = True, use_api: bool = False):
    from rag_finance_system.src.document_processor import DocumentProcessor
    from rag_finance_system.src.embedder import Embedder, Reranker
    from rag_finance_system.src.vector_store import VectorStore
    from rag_finance_system.src.retriever import Retriever
    from rag_finance_system.src.llm import get_llm
    from rag_finance_system.src.rag_chain import RAGChain
    from rag_finance_system.src.rewriter import QueryRewriter

    embedder = Embedder()
    vector_store = VectorStore()

    reranker = None
    if use_reranker:
        try:
            reranker = Reranker()
        except Exception as e:
            st.warning(f"Reranker加载失败: {e}")

    # 查询重写小模型（优先加载微调权重，失败回退大模型）
    rewriter = None
    try:
        lora_path = str(Path(__file__).resolve().parent.parent.parent / "checkpoints" / "rewriter_lora" / "final")
        if Path(lora_path).exists():
            rewriter = QueryRewriter(model_path="Qwen/Qwen2.5-0.5B-Instruct", lora_path=lora_path)
        else:
            rewriter = QueryRewriter()
    except Exception as e:
        st.warning(f"查询重写小模型加载失败: {e}，将使用主LLM重写")

    llm = get_llm(prefer_local=not use_api)
    retriever = Retriever(embedder=embedder, vector_store=vector_store, reranker=reranker)
    rag = RAGChain(retriever=retriever, llm=llm, rewriter=rewriter)
    processor = DocumentProcessor()

    return {
        "processor": processor,
        "embedder": embedder,
        "vector_store": vector_store,
        "rag": rag,
    }


# ===== 侧边栏 =====
with st.sidebar:
    st.title("⚙️ 系统设置")

    use_api = st.toggle("使用API模式（通义千问）", value=False,
                        help="显存不足时开启，需要在.env配置DASHSCOPE_API_KEY")
    use_reranker = st.toggle("启用Reranker精排", value=True,
                             help="提升检索精度，需额外~1.1GB显存")
    use_query_rewrite = st.toggle("启用查询重写", value=True,
                                  help="将用户问题改写为更适合向量检索的查询，提高检索召回效果")

    mode = st.selectbox(
        "检索模式",
        ["全部", "仅法条", "仅案例", "仅其他"],
        index=0,
        help="选择本次问答检索范围：全部/仅法规/仅案例/仅其他资料",
    )

    st.divider()
    st.subheader("文档管理")

    uploaded_file = st.file_uploader(
        "上传金融法规PDF或TXT",
        type=["pdf", "txt"],
        help="支持中文PDF或纯文本TXT，最大200MB"
    )
    
    case_file = st.file_uploader(
        "上传案例文件",
        type=["pdf","txt"],
        help="支持中文PDF或纯文本TXT，最大200MB"
    )

    other_file = st.file_uploader(
        "上传其它参考资料",
        type=["pdf","txt"],
        help="支持PDF或TXT，如学术文献、研究报告、政策解读等，最大200MB"
    )

    if other_file:
        if st.button("解析资料并建立索引", type="secondary", key="other_btn"):

            save_path = Path("data/raw/other") / other_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(other_file.getvalue())

            with st.spinner(f"正在解析资料 {other_file.name}..."):
                comps = load_components(use_reranker, use_api)

                chunks = comps["processor"].process_file(str(save_path), doc_type="other")

                for c in chunks:
                    c["source"] = c.get("source", other_file.name)

                texts = [c["text"] for c in chunks]

                embeddings = comps["embedder"].encode_documents(texts)

                comps["vector_store"].insert(chunks, embeddings)

                st.success(f"✅ 资料入库完成：{len(chunks)} 条")

    if case_file:
        if st.button("解析案例并建立索引", type="secondary", key="case_btn"):

            save_path = Path("data/raw/case") / case_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                f.write(case_file.getvalue())

            with st.spinner(f"正在解析案例 {case_file.name}..."):
                comps = load_components(use_reranker, use_api)

                chunks = comps["processor"].process_file(str(save_path), doc_type="case")

                # 👇 核心区别就在这
                for c in chunks:
                    c["source"] = "case"

                texts = [c["text"] for c in chunks]

                embeddings = comps["embedder"].encode_documents(texts)

                comps["vector_store"].insert(chunks, embeddings)

                st.success(f"✅ 案例入库完成：{len(chunks)} 条")

    if uploaded_file:
        if st.button("解析并建立索引", type="primary"):
            # 保存上传文件
            save_path = Path("data/raw") / uploaded_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # 建立索引
            with st.spinner(f"正在解析 {uploaded_file.name}..."):
                comps = load_components(use_reranker, use_api)
                chunks = comps["processor"].process_file(str(save_path), doc_type="law")
                texts = [c["text"] for c in chunks]

                progress_bar = st.progress(0, text="生成向量...")
                batch_size = 16
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    embs = comps["embedder"].encode_documents(batch)
                    all_embeddings.extend(embs)
                    progress_bar.progress(
                        min((i + batch_size) / len(texts), 1.0),
                        text=f"生成向量... {min(i + batch_size, len(texts))}/{len(texts)}"
                    )

                comps["vector_store"].insert(chunks, all_embeddings)
                st.success(f"✅ 索引建立完成！共 {len(chunks)} 个知识片段")

    st.divider()
    st.subheader("批量导入")
    if st.button("一键导入 testfiles 中的 148 份其它资料", type="secondary", key="batch_import"):
        import glob as _glob
        _txt_dir = Path("data/testfiles")
        _txt_files = sorted(_glob.glob(str(_txt_dir / "**" / "*.txt"), recursive=True))
        if not _txt_files:
            st.warning("未找到 .txt 文件，请先运行 tools/convert_testfiles.py 转换")
        else:
            comps = load_components(use_reranker, use_api)
            progress = st.progress(0, text=f"0/{len(_txt_files)}")
            total_chunks = 0
            for i, fp in enumerate(_txt_files):
                try:
                    chunks = comps["processor"].process_file(fp, doc_type="other")
                    texts = [c["text"] for c in chunks]
                    embeddings = comps["embedder"].encode_documents(texts)
                    comps["vector_store"].insert(chunks, embeddings)
                    total_chunks += len(chunks)
                except Exception as e:
                    st.warning(f"跳过 {Path(fp).name}: {e}")
                progress.progress((i + 1) / len(_txt_files),
                                  text=f"{i + 1}/{len(_txt_files)}  ({total_chunks} chunks)")
            st.success(f"✅ 批量导入完成！{len(_txt_files)} 个文件，共 {total_chunks} 个 chunk")

    st.divider()
    st.caption("Phase 1 最小闭环版本\n向量检索 + Reranker + Qwen2.5")


# ===== 主界面 =====
st.title("📚 金融制度知识问答系统")
st.caption("基于RAG的金融法规智能问答 | bge-large-zh-v1.5 + Qwen2.5-7B")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 查看溯源条文 ({len(msg['sources'])} 条)"):
                for i, src in enumerate(msg["sources"], 1):
                    conf_color = "green" if src["score"] > 0.7 else "orange" if src["score"] > 0.4 else "red"
                    st.markdown(
                        f"**[{i}] 【{src['source']} {src['article_num']}】** "
                        f":{conf_color}[相关度 {src['score']:.2%}]"
                    )
                    st.text(src["text"][:200] + "..." if len(src["text"]) > 200 else src["text"])
                    if i < len(msg["sources"]):
                        st.divider()

        if msg.get("confidence"):
            conf = msg["confidence"]
            total = conf["total"]
            col1, col2, col3 = st.columns(3)
            col1.metric("综合可信度", f"{total:.1%}")
            col2.metric("检索相关性", f"{conf['retrieval']:.1%}")
            col3.metric("答案覆盖度", f"{conf['coverage']:.1%}")


# 用户输入
if question := st.chat_input("请输入您的金融法规问题..."):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("检索相关条文并生成答案..."):
            try:
                comps = load_components(use_reranker, use_api)
                doc_type_filter = None
                if mode == "仅法条":
                    doc_type_filter = "law"
                elif mode == "仅案例":
                    doc_type_filter = "case"
                elif mode == "仅其他":
                    doc_type_filter = "other"
                result = comps["rag"].query(
                    question,
                    doc_type_filter=doc_type_filter,
                    use_query_rewrite=use_query_rewrite,
                )

                answer = result["answer"]
                sources = result["sources"]
                confidence = result["confidence"]
                rewritten_query = result.get("rewritten_query", question)

                st.markdown(answer)
                if rewritten_query and rewritten_query != question:
                    st.caption(f"已改写查询：{rewritten_query}")

                # 溯源展示
                if sources:
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

                # 可信度
                total = confidence["total"]
                col1, col2, col3 = st.columns(3)
                col1.metric("综合可信度", f"{total:.1%}")
                col2.metric("检索相关性", f"{confidence['retrieval']:.1%}")
                col3.metric("答案覆盖度", f"{confidence['coverage']:.1%}")

                # 保存到历史
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "confidence": confidence,
                })

            except Exception as e:
                err_msg = f"出错了: {str(e)}\n\n请检查：\n1. 是否已上传PDF并建立索引\n2. 模型路径是否正确\n3. 显存是否充足"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
