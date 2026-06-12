"""
test_retrieval_baseline.py
纯检索基线脚本 — 不依赖 LLM，只测 parse→embed→insert→search 闭环。
用法: py -3 rag_finance_system/test_retrieval_baseline.py [--file <txt_path>] [--keep-collection]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from rag_finance_system.src.document_processor import DocumentProcessor
from rag_finance_system.src.embedder import Embedder
from rag_finance_system.src.vector_store import VectorStore

BASELINE_COLLECTION = "baseline_test"

DEFAULT_FILE = str(
    Path(__file__).parent
    / "src"
    / "txt_files"
    / "中华人民共和国公司法_20231229.txt"
)


def override_collection_name(vs: "VectorStore") -> None:
    vs.collection_name = BASELINE_COLLECTION
    vs._index_type = None
    if vs.client.has_collection(BASELINE_COLLECTION):
        vs.client.load_collection(BASELINE_COLLECTION)
        vs._index_type = vs._detect_index_type()


def cleanup(vs: "VectorStore") -> None:
    if vs.client.has_collection(BASELINE_COLLECTION):
        vs.client.release_collection(BASELINE_COLLECTION)
        vs.client.drop_collection(BASELINE_COLLECTION)
        logger.info(f"已清理基线 collection: {BASELINE_COLLECTION}")


def run_baseline(file_path: str, keep: bool = False) -> dict:
    results: dict = {"passed": [], "failed": [], "skipped": []}

    # ---- Step 1: Parse ----
    logger.info("===== [1/5] 文档解析 =====")
    processor = DocumentProcessor()
    try:
        chunks = processor.process_file(file_path)
    except Exception as e:
        logger.error(f"解析失败: {e}")
        results["failed"].append(f"parse: {e}")
        return results

    if not chunks:
        results["failed"].append("parse: 解析结果为空")
        return results

    results["passed"].append(f"parse: {len(chunks)} chunks")
    logger.info(f"解析: {len(chunks)} chunks")

    # ---- Step 2: Embed ----
    logger.info("===== [2/5] Embedding =====")
    try:
        embedder = Embedder()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode_documents(texts, batch_size=16)
    except Exception as e:
        logger.error(f"Embedding失败: {e}")
        results["failed"].append(f"embed: {e}")
        return results

    if len(embeddings) != len(chunks):
        results["failed"].append(
            f"embed: 向量数({len(embeddings)}) != chunk数({len(chunks)})"
        )
        return results

    dim = len(embeddings[0])
    results["passed"].append(f"embed: {len(embeddings)} vectors, dim={dim}")
    logger.info(f"Embedding: {len(embeddings)} 向量, dim={dim}")

    # ---- Step 3: Insert ----
    logger.info("===== [3/5] 写入 Milvus =====")
    try:
        vs = VectorStore()
        override_collection_name(vs)
        inserted = vs.insert(chunks, embeddings)
    except Exception as e:
        logger.error(f"写入失败: {e}")
        results["failed"].append(f"insert: {e}")
        return results

    if inserted != len(chunks):
        results["failed"].append(
            f"insert: 写入{inserted} != 预期{len(chunks)}"
        )
        cleanup(vs)
        return results

    results["passed"].append(f"insert: {inserted} rows")
    logger.info(f"写入: {inserted} rows")

    # verify stats
    stats = vs.get_collection_stats()
    row_count = stats.get("count", stats.get("row_count", 0))
    if row_count != len(chunks):
        results["failed"].append(
            f"stats: row_count={row_count}, 预期={len(chunks)}"
        )
        cleanup(vs)
        return results
    results["passed"].append(f"stats: {row_count} rows")

    # verify index_type detection
    idx_type = vs._detect_index_type()
    if idx_type:
        results["passed"].append(f"index_type: {idx_type}")
    else:
        results["failed"].append("index_type: 检测返回 None/空")

    # ---- Step 4: Search ----
    logger.info("===== [4/5] 向量检索 =====")
    query = "股东责任是什么？"
    try:
        query_vec = embedder.encode_query(query)
        hits = vs.search(query_vec, top_k=5)
    except Exception as e:
        logger.error(f"检索失败: {e}")
        results["failed"].append(f"search: {e}")
        cleanup(vs)
        return results

    if not hits:
        results["failed"].append("search: 无结果返回")
        cleanup(vs)
        return results

    results["passed"].append(f"search: {len(hits)} hits for '{query}'")

    # verify scores in [0,1]
    bad_scores = [h for h in hits if not (0.0 <= h["score"] <= 1.0)]
    if bad_scores:
        results["failed"].append(
            f"score_range: {len(bad_scores)} hits 分数越界 {[h['score'] for h in bad_scores[:3]]}"
        )
    else:
        results["passed"].append("score_range: all in [0,1]")

    # print top-3 hits for manual review
    for i, h in enumerate(hits[:3], 1):
        logger.info(
            f"  [{i}] score={h['score']:.4f} "
            f"law={h.get('law_name','?')[:30]} "
            f"text={h['text'][:80]}..."
        )

    # ---- Step 5: Filter search ----
    logger.info("===== [5/5] 标量过滤检索 =====")
    law_name = chunks[0].get("law_name", "")
    if law_name:
        try:
            filtered = vs.search(query_vec, top_k=5, law_name_filter=law_name)
        except Exception as e:
            logger.error(f"过滤检索失败: {e}")
            results["failed"].append(f"filter_search: {e}")
            cleanup(vs)
            return results

        if not filtered:
            results["failed"].append("filter_search: 过滤后无结果")
        else:
            results["passed"].append(f"filter_search: {len(filtered)} hits")
            mismatched = [
                h for h in filtered if h.get("law_name", "") != law_name
            ]
            if mismatched:
                results["failed"].append(
                    f"filter_accuracy: {len(mismatched)}/{len(filtered)} 不匹配过滤条件"
                )
            else:
                results["passed"].append("filter_accuracy: 100% 匹配")
    else:
        results["skipped"].append("filter_search: chunks 无 law_name 字段")

    # ---- Cleanup ----
    if keep:
        logger.info(f"保留 collection: {BASELINE_COLLECTION}")
    else:
        cleanup(vs)

    return results


def main():
    parser = argparse.ArgumentParser(description="纯检索基线测试 (无LLM依赖)")
    parser.add_argument("--file", type=str, default=DEFAULT_FILE,
                        help="用于测试的TXT文件路径")
    parser.add_argument("--keep-collection", action="store_true",
                        help="测试后保留 Milvus collection")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"文件不存在: {args.file}")
        sys.exit(1)

    logger.info(f"基线文件: {args.file}")
    results = run_baseline(args.file, keep=args.keep_collection)

    print("\n" + "=" * 60)
    passed = len(results["passed"])
    failed = len(results["failed"])
    skipped = len(results.get("skipped", []))
    total = passed + failed + skipped

    print(f"基线结果: {passed} passed, {failed} failed, {skipped} skipped ({total} total)")
    print("=" * 60)

    for item in results["passed"]:
        print(f"  PASS  {item}")
    for item in results.get("skipped", []):
        print(f"  SKIP  {item}")
    for item in results["failed"]:
        print(f"  FAIL  {item}")

    print("=" * 60)

    if failed:
        print("BASELINE FAILED")
        sys.exit(1)
    else:
        print("BASELINE PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
