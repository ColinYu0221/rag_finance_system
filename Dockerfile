# ============================================================
# 金融法规 RAG 系统 — GPU Docker 镜像
# 基础: PyTorch 2.11 + CUDA 12.6 + cuDNN 9
# 模型: Qwen2.5-7B-Instruct-GPTQ-Int4 (构建时下载)
# ============================================================

FROM docker.m.daocloud.io/pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

# ── 系统依赖 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# ── Python 依赖（利用 Docker 缓存层） ──
COPY docker/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ── 构建时下载 Embedding + Reranker 模型 ──
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('BAAI/bge-small-zh-v1.5', \
    local_dir='/app/models/bge-small-zh-v1.5', \
    local_dir_use_symlinks=False, \
    resume_download=True); \
snapshot_download('BAAI/bge-reranker-v2-m3', \
    local_dir='/app/models/bge-reranker-v2-m3', \
    local_dir_use_symlinks=False, \
    resume_download=True)"

# ── 构建时下载 Qwen2.5-7B-Instruct-GPTQ-Int4（约 5.6GB） ──
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', \
    local_dir='/app/models/Qwen2.5-7B-Int4', \
    local_dir_use_symlinks=False, \
    resume_download=True)"

# ── 复制源码 + 数据 ──
COPY rag_finance_system/ /app/rag_finance_system/
COPY data/ /app/data/
COPY scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh && \
    sed -i 's/\r$//' /app/docker-entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
