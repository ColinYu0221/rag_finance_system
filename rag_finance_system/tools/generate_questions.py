"""
读取 testfiles 下的所有 .txt 文件，调用 DeepSeek API 生成约 300 条问题，
保存到 questions.json 文件中，格式为 {"question": "问题内容"}。
"""

import os
import json
import glob
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

TESTFILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "testfiles")
)
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "questions.json")
TARGET_QUESTIONS = 300
BATCH_SIZE = 6  # 每批次处理的文件数


def read_all_txt_files():
    """读取所有 .txt 文件，返回 [(文件名, 内容), ...]"""
    pattern = os.path.join(TESTFILES_DIR, "**", "*.txt")
    files = glob.glob(pattern, recursive=True)
    results = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if content:
            rel_path = os.path.relpath(fp, TESTFILES_DIR)
            results.append((rel_path, content))
    return results


def generate_questions_batch(file_batch, target_count):
    """
    传入一批文件（文件名+内容），调用 LLM 生成 target_count 条问题。
    """
    documents_text = ""
    for i, (fname, content) in enumerate(file_batch, 1):
        # 每篇文档截断至 1500 字，避免 token 超限
        truncated = content[:1800]
        documents_text += f"\n--- 文档{i}: {fname} ---\n{truncated}\n"

    prompt = f"""你是一个金融监管领域的专家。以下是一批中国金融监督管理局（银保监局/保监局）发布的规范性文件。
请根据这些文件的内容，生成 {target_count} 条高质量的检索式问答问题。

要求：
1. 问题必须基于文档的具体内容，不能凭空编造
2. 问题类型要多样化：包括事实查询、政策解读、合规要求、适用范围、时间节点、监管标准、操作流程等
3. 问题表述要自然，像是真实用户在查询时的提问方式
4. 问题长度适中（15-50字），语义清晰
5. 每条问题独占一行，以数字序号开头（如"1. "、"2. "）
6. 只输出问题列表，不要任何解释或额外内容

{documents_text}

请生成 {target_count} 条问题："""

    response = client.chat.completions.create(
        model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.8,  # 稍微提高温度以增加多样性
    )
    return response.choices[0].message.content


def parse_questions(raw_text):
    """从 LLM 返回的文本中解析出问题列表"""
    questions = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # 去除序号前缀 "1. " "1、" "1）" 等
        for prefix_sep in [". ", ".", "、", "）", ") ", ")"]:
            # 找到第一个空格/分隔符位置
            idx_combined = line.find(prefix_sep)
            if idx_combined > 0 and idx_combined < 12:
                prefix_part = line[:idx_combined]
                # 检查前缀是否是数字
                prefix_num = prefix_part.strip()
                try:
                    int(prefix_num)
                    line = line[idx_combined + len(prefix_sep):].strip()
                    break
                except ValueError:
                    pass
        if line and len(line) >= 8:  # 问题至少8个字
            questions.append(line)
    return questions


def main():
    print(f"读取 {TESTFILES_DIR} 下的所有 .txt 文件...")
    all_files = read_all_txt_files()
    print(f"共读取 {len(all_files)} 个有效文件")

    # 计算每批需要生成的问题数
    num_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE
    questions_per_batch = TARGET_QUESTIONS // num_batches
    remainder = TARGET_QUESTIONS % num_batches

    all_questions = []
    batch_sizes = [questions_per_batch + 1 if i < remainder else questions_per_batch
                   for i in range(num_batches)]

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(all_files))
        batch = all_files[start:end]
        target = batch_sizes[batch_idx]

        print(f"\n批次 {batch_idx + 1}/{num_batches}: 文件 {start + 1}-{end}, 目标生成 {target} 条问题...")

        retry = 0
        while retry < 3:
            try:
                raw_output = generate_questions_batch(batch, target)
                questions = parse_questions(raw_output)
                if len(questions) >= target * 0.6:  # 至少生成目标的60%
                    all_questions.extend(questions)
                    print(f"  生成 {len(questions)} 条有效问题")
                    break
                else:
                    retry += 1
                    print(f"  生成不足 ({len(questions)} < {target * 0.6:.0f})，重试 {retry}/3")
            except Exception as e:
                retry += 1
                print(f"  调用失败: {e}，重试 {retry}/3")

        if retry == 3:
            print(f"  批次失败，已跳过")

    # 去重
    seen = set()
    unique_questions = []
    for q in all_questions:
        if q not in seen:
            seen.add(q)
            unique_questions.append(q)

    # 限制到目标数量左右
    if len(unique_questions) > TARGET_QUESTIONS:
        unique_questions = unique_questions[:TARGET_QUESTIONS]

    # 保存为 JSON 格式
    output = [{"question": q} for q in unique_questions]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n完成！共生成 {len(unique_questions)} 条问题，保存到 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
