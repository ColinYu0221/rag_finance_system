"""
下载 README 要求的必备模型到 models/ 目录
- BAAI/bge-small-zh-v1.5  ->  models/bge-small-zh-v1.5
- BAAI/bge-reranker-v2-m3 ->  models/bge-reranker-v2-m3
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download
from pathlib import Path
import time

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

models = [
    ("BAAI/bge-small-zh-v1.5",  MODELS_DIR / "bge-small-zh-v1.5"),
    ("BAAI/bge-reranker-v2-m3", MODELS_DIR / "bge-reranker-v2-m3"),
]

for repo_id, local_dir in models:
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[skip] {local_dir.name} already exists")
        continue
    print(f"downloading {repo_id} -> {local_dir} ...")
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    elapsed = time.time() - t0
    print(f"[done] ({elapsed:.1f}s)")

print("\nAll models downloaded!")
