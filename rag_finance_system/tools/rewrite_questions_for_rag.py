"""
读取 questions.json，调用 DeepSeek API 对每条问题进行 RAG 查询重写，
将重写后的 query 添加到每条记录中，格式为 {"question": "...", "query": "..."}
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com"),
)

INPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "questions.json")
BATCH_SIZE = 20  # 每批处理的问题数


def rewrite_batch(questions, start_idx):
    """
    对一批问题进行 RAG 查询重写。
    RAG 重写目标：将自然语言问题转换为更适合向量检索的查询语句。
    包括：关键词提取、同义词扩展、去除口语化表达、补充专业术语。
    """
    questions_text = ""
    for i, q in enumerate(questions, 1):
        questions_text += f"{i}. {q}\n"

    prompt = f"""你是一个 RAG（检索增强生成）系统的查询优化专家。以下是{len(questions)}条来自金融监管领域的自然语言问题。

请对每条问题进行查询重写（Query Rewriting），使得重写后的查询更适合向量检索。
重写策略：
1. 提取核心关键词和专业术语
2. 补充同义词和相关表述（例如"银保监局"可补充"金融监督管理局"）
3. 去除口语化语气词，保留查询意图
4. 将简写/缩写展开为完整表述
5. 保持重写后的查询为一句通顺的话，长度在20-60字之间
6. 不要改变原问题的核心意图

请严格按照以下格式输出，每条一行，以序号开头：
1. [重写后的查询]
2. [重写后的查询]
...

原始问题：
{questions_text}

重写后的查询："""

    response = client.chat.completions.create(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.3,  # 低温度确保重写的一致性和准确性
    )
    return response.choices[0].message.content


def parse_rewrites(raw_text, expected_count):
    """从 LLM 返回中解析重写后的查询"""
    rewrites = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for sep in [". ", ".", "、", "）", ") ", ")"]:
            idx = line.find(sep)
            if 0 < idx < 8:
                prefix = line[:idx].strip()
                try:
                    int(prefix)
                    line = line[idx + len(sep):].strip()
                    break
                except ValueError:
                    pass
        if line and len(line) >= 8:
            rewrites.append(line)
    # 确保数量匹配
    while len(rewrites) < expected_count:
        rewrites.append(rewrites[-1] if rewrites else "")
    return rewrites[:expected_count]


def main():
    print(f"读取 {INPUT_FILE} ...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [item["question"] for item in data]
    total = len(questions)
    print(f"共 {total} 条问题，开始批量重写...")

    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    all_rewrites = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, total)
        batch = questions[start:end]

        print(f"批次 {batch_idx + 1}/{num_batches}: 问题 {start + 1}-{end} ...", end=" ")

        retry = 0
        success = False
        while retry < 3:
            try:
                raw = rewrite_batch(batch, start)
                rewrites = parse_rewrites(raw, len(batch))
                if len(rewrites) == len(batch):
                    all_rewrites.extend(rewrites)
                    print(f"完成 ({len(rewrites)} 条)")
                    success = True
                    break
                else:
                    retry += 1
                    print(f"数量不匹配 ({len(rewrites)} vs {len(batch)})，重试 {retry}/3")
            except Exception as e:
                retry += 1
                print(f"失败: {e}，重试 {retry}/3")

        if not success:
            print("批次失败，使用原始问题作为查询")
            all_rewrites.extend(batch)

    # 组装输出，query 放在 question 后面
    output = []
    for item, rewrite in zip(data, all_rewrites):
        output.append({
            "question": item["question"],
            "query": rewrite,
        })

    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完成！已更新 {len(output)} 条记录，保存到 {INPUT_FILE}")


if __name__ == "__main__":
    main()
