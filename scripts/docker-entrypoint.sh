#!/bin/bash
set -e

cd /app/rag_finance_system

echo "========================================="
echo "  金融法规 RAG 系统 — Docker 启动"
echo "========================================="

# ── 1. 初始化 MySQL 表结构 ──
echo ""
echo "=== [1/3] 初始化 MySQL 表结构 ==="
python -c "import src.models; from src.database import init_db; init_db()" \
    && echo "[OK] MySQL 表创建成功" \
    || echo "[WARN] MySQL 不可用，对话历史功能将降级"

# ── 2. 启动 FastAPI 后台进程 ──
echo ""
echo "=== [2/3] 启动 FastAPI 后端 ==="
uvicorn rag_finance_system.api_app:app \
    --host 0.0.0.0 --port 8000 \
    --log-level info &
API_PID=$!

echo "等待 API 就绪 (PID: $API_PID)..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/openapi.json > /dev/null 2>&1; then
        echo "[OK] FastAPI 就绪"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "[ERROR] API 启动超时"
        exit 1
    fi
    sleep 2
done

# ── 3. 启动 Streamlit 前端 ──
echo ""
echo "=== [3/3] 启动 Streamlit 前端 ==="
echo "前端地址: http://0.0.0.0:8501"
exec streamlit run rag_finance_system/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
