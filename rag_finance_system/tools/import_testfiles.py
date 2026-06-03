"""
一次性脚本：将 data/testfiles/ 下所有 .txt 文件导入向量库（doc_type="other"）。
用法：python tools/import_testfiles.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag_finance_system.src.document_processor import DocumentProcessor
from rag_finance_system.src.embedder import Embedder
from rag_finance_system.src.vector_store import VectorStore

BASE = Path(__file__).resolve().parent.parent.parent / "data" / "testfiles"

processor = DocumentProcessor()
embedder = Embedder()
store = VectorStore()

files = sorted(BASE.rglob("*.txt"))
print(f"找到 {len(files)} 个 .txt 文件")

total = 0
for f in files:
    try:
        chunks = processor.process_file(str(f), doc_type="other")
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_documents(texts)
        n = store.insert(chunks, embeddings)
        total += n
        print(f"  OK  {f.name}  ({n} chunks)")
    except Exception as e:
        print(f"  FAIL  {f.name}: {e}")

print(f"\n完成。共入库 {total} 个 chunk")
