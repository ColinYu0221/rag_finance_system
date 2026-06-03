"""
generate_rewrite_data.py
使用 LLM API 批量生成查询重写训练数据（question → rewritten_query 对）

两种模式：
  模式一（正向）：给定自然语言问题列表，用 LLM 改写为检索查询
    输入：questions.txt（一行一个问题）
    输出：rewrite_pairs.jsonl

  模式二（反向）：给定文档 chunks，LLM 先提取检索关键词，再反向生成口语问题
    输入：chunks 目录下的 .txt 文件（每行一个 chunk）
    输出：rewrite_pairs.jsonl

用法：
  # 模式一：从问题列表生成
  python tools/generate_rewrite_data.py \
    --mode forward \
    --input data/questions.txt \
    --output data/rewrite_pairs.jsonl

  # 模式二：从文档 chunks 反向生成
  python tools/generate_rewrite_data.py \
    --mode reverse \
    --input data/chunks/ \
    --output data/rewrite_pairs.jsonl \
    --num-questions 1000

  # 通用参数
  --api {deepseek,qwen}    选择 API（默认 deepseek）
  --model MODEL            模型名（deepseek 默认 deepseek-chat）
  --batch-size N           每批生成数量（默认 20）
  --resume                  从已有输出文件断点续传
  --temperature FLOAT      生成温度（默认 0.3，正向模式稍高增加多样性）
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# 自动查找项目根目录下的 .env（向上两级：tools/ → rag_finance_system/）
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ========================================================================
# Prompt 模板
# ========================================================================

FORWARD_SYSTEM_PROMPT = """你是一个中文检索查询改写助手。
你的任务是将用户的自然语言问题改写为一条简洁、精准、适合向量检索的查询语句。
请保留关键实体和意图，去掉无关废话。

重要规则：
1. 仅输出改写后的查询，一行一个
2. 不要输出解释、编号、前缀或格式说明
3. 例如输入"公司破产了股东要承担什么责任" → 输出"公司破产 股东责任"
4. 保留金融法律术语的精确性（如"资本充足率""反洗钱""连带责任"）"""

REVERSE_SYSTEM_PROMPT = """你是一个中文训练数据生成助手。
给定一条金融法规或案例文书片段，请完成两步：

步骤一：提取该片段中的核心检索关键词或查询短语（适合向量检索的短句）
步骤二：反向生成一个用户可能会用来查询该条文的自然语言口语问题

输出格式（严格 JSON，每条片段一行）：
{"query": "检索查询短语", "question": "用户口语问题"}

要求：
1. question 应模拟真实用户的口语表达，包含疑问语气
2. query 应简洁精准，保留关键金融法律术语
3. question 与 query 的语义必须一致
4. 至少生成 3 个不同的 (query, question) 对，选用不同角度提问
5. 问题类型多样化：定义类（"什么是..."）、适用类（"...适用于哪些情况"）、计算类（"...如何计算"）、案例类（"...是否会构成..."）
6. 只输出 JSON Lines，不要输出任何其他内容"""


# ========================================================================
# API 客户端
# ========================================================================

def build_client(api: str) -> tuple[OpenAI, str]:
    if api == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("请在 .env 中设置 DEEPSEEK_API_KEY")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base = os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
        return OpenAI(api_key=key, base_url=base), model
    elif api == "qwen":
        key = os.getenv("DASHSCOPE_API_KEY", "")
        if not key:
            raise ValueError("请在 .env 中设置 DASHSCOPE_API_KEY")
        return (
            OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "qwen-plus",
        )
    else:
        raise ValueError(f"不支持的 API: {api}")


def call_llm(client: OpenAI, model: str, system: str, user: str,
             temperature: float = 0.1, max_tokens: int = 512) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ========================================================================
# 已处理记录管理（断点续传）
# ========================================================================

def load_done_keys(output_path: str) -> set[str]:
    done: set[str] = set()
    out = Path(output_path)
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(obj.get("question", ""))
                except json.JSONDecodeError:
                    continue
    return done


# ========================================================================
# 模式一：正向生成（问题 → 检索查询）
# ========================================================================

def forward_generate(args):
    """从问题列表出发，LLM 改写成检索查询。"""
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    questions = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    logger.info(f"读取到 {len(questions)} 个问题")

    client, model = build_client(args.api)
    done = load_done_keys(args.output) if args.resume else set()
    pending = [q for q in questions if q not in done]

    if args.resume and done:
        logger.info(f"跳过已完成的 {len(done)} 条，剩余 {len(pending)} 条")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    total_generated = 0

    for i in range(0, len(pending), batch_size):
        batch = pending[i:i + batch_size]

        # 构造批量改写 prompt
        questions_text = "\n".join(f"{j+1}. {q}" for j, q in enumerate(batch))
        user_message = (
            f"请将以下 {len(batch)} 条自然语言问题逐一改写为适合向量检索的简洁查询。\n"
            f"每行输出一条改写结果（不要编号）：\n\n{questions_text}"
        )

        try:
            result = call_llm(client, model, FORWARD_SYSTEM_PROMPT, user_message,
                              temperature=args.temperature, max_tokens=1024)
            rewritten = [line.strip() for line in result.split("\n") if line.strip()]

            if len(rewritten) != len(batch):
                logger.warning(f"第 {i+1}-{i+len(batch)} 条：期望 {len(batch)} 条输出，"
                               f"实际 {len(rewritten)} 条，跳过本轮")
                continue

            # 写入 JSONL
            with out_path.open("a", encoding="utf-8") as f:
                for question, query in zip(batch, rewritten):
                    obj = {"question": question, "query": query}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            total_generated += len(batch)
            logger.info(f"进度: {min(i + batch_size, len(pending))}/{len(pending)}")

        except Exception as e:
            logger.error(f"第 {i+1}-{i+len(batch)} 条失败: {e}")
            logger.info(f"可重新运行并加 --resume 继续")
            break

        time.sleep(0.3)  # 降低 API 速率压力

    logger.info(f"完成。生成了 {total_generated} 条，输出到 {args.output}")


# ========================================================================
# 模式二：反向生成（chunk → query + question）
# ========================================================================

def load_chunks(input_dir: str, max_chars: int = 300) -> list[str]:
    """从目录加载文本片段。"""
    chunks = []
    dir_path = Path(input_dir)
    for txt_file in dir_path.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        # 按空行或短句拆成片段
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 30]
        chunks.extend(paragraphs)

    # 截取前 max_chars 字符，避免 token 超限
    chunks = [c[:max_chars] for c in chunks if len(c) >= 20]
    return chunks


def reverse_generate(args):
    """从文档 chunks 反向生成 (query, question) 训练对。"""
    chunks = load_chunks(args.input)
    logger.info(f"读取到 {len(chunks)} 个文本片段")

    if args.num_questions:
        import random
        random.seed(42)
        chunks = random.sample(chunks, min(args.num_questions // 3, len(chunks)))
        logger.info(f"采样后 {len(chunks)} 个片段（目标约 {args.num_questions} 条训练数据）")

    client, model = build_client(args.api)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.output) if args.resume else set()

    batch_size = args.batch_size
    total_generated = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        # 每个 chunk 要求生成 3 个 (query, question) 对
        chunks_text = "\n\n".join(f"[{j+1}] {c}" for j, c in enumerate(batch))
        user_message = (
            f"请为以下 {len(batch)} 条法规或案例片段各生成至少 3 个不同的 (query, question) 训练对。\n\n"
            f"{chunks_text}\n\n"
            f"输出严格 JSON Lines，每行一个对象，格式：\n"
            f'{{"query": "检索查询", "question": "用户口语问题"}}'
        )

        try:
            result = call_llm(client, model, REVERSE_SYSTEM_PROMPT, user_message,
                              temperature=args.temperature, max_tokens=4096)

            # 解析 JSONL 输出
            parsed = 0
            with out_path.open("a", encoding="utf-8") as f:
                for line in result.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        q = obj.get("question", "")
                        if q and q not in done and "query" in obj and obj["query"]:
                            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                            done.add(q)
                            parsed += 1
                    except json.JSONDecodeError:
                        continue

            total_generated += parsed
            logger.info(f"进度: {min(i + batch_size, len(chunks))}/{len(chunks)} "
                        f"(本批产出 {parsed} 条)")

        except Exception as e:
            logger.error(f"第 {i+1}-{i+len(batch)} 条失败: {e}")
            logger.info(f"可重新运行并加 --resume 继续")
            break

        time.sleep(0.5)

    logger.info(f"完成。共生成 {total_generated} 条训练数据，输出到 {args.output}")


# ========================================================================
# CLI
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="使用 LLM API 生成查询重写训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 正向：从问题列表生成
  python tools/generate_rewrite_data.py -m forward -i data/questions.txt -o data/forward_pairs.jsonl

  # 反向：从文档 chunks 生成
  python tools/generate_rewrite_data.py -m reverse -i data/raw/ -o data/reverse_pairs.jsonl -n 500

  # 断点续传
  python tools/generate_rewrite_data.py -m forward -i data/questions.txt -o data/pairs.jsonl --resume
        """,
    )
    parser.add_argument("-m", "--mode", choices=["forward", "reverse"], required=True,
                        help="生成模式：forward=问题→查询, reverse=chunk→(查询+问题)")
    parser.add_argument("-i", "--input", required=True,
                        help="输入路径：forward 模式为 questions.txt，reverse 模式为 chunks 目录")
    parser.add_argument("-o", "--output", default="data/rewrite_pairs.jsonl",
                        help="输出 JSONL 文件路径（默认 data/rewrite_pairs.jsonl）")
    parser.add_argument("-n", "--num-questions", type=int, default=None,
                        help="reverse 模式下目标训练数据总量")
    parser.add_argument("--api", choices=["deepseek", "qwen"], default="deepseek",
                        help="使用的 LLM API（默认 deepseek）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名（deepseek 默认 deepseek-chat，qwen 默认 qwen-plus）")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="每批处理的条数（默认 20）")
    parser.add_argument("--resume", action="store_true",
                        help="断点续传，跳过输出文件中已有的 question")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="生成温度（默认 0.3）")
    args = parser.parse_args()

    if args.mode == "forward":
        forward_generate(args)
    else:
        reverse_generate(args)


if __name__ == "__main__":
    main()
