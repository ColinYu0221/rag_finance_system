"""
train_rewriter.py
LoRA 微调查询重写小模型（Qwen2.5-0.5B-Instruct）。
输入：data/questions.json  {"question": "...", "query": "..."}
输出：checkpoints/rewriter_lora/
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ===== 配置 =====
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "questions.json"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "rewriter_lora"
TRAIN_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 256

SYSTEM_PROMPT = "将以下问题改写为适合向量检索的简洁查询，只输出改写后的查询。"


def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 去重，确保 query 不为空
    seen = set()
    pairs = []
    for item in raw:
        q, r = item.get("question", ""), item.get("query", "")
        if q and r and q not in seen:
            seen.add(q)
            pairs.append(item)
    print(f"加载 {len(pairs)} 条训练对（去重后，原始 {len(raw)} 条）")

    # 拆训练/验证集
    split = int(len(pairs) * 0.9)
    return Dataset.from_list(pairs[:split]), Dataset.from_list(pairs[split:])


def format_example(question, query):
    """构建 ChatML 格式的输入和标签。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{question}"},
        {"role": "assistant", "content": query},
    ]
    return {"text": messages}


def tokenize(examples, tokenizer):
    """Tokenize 为训练格式。"""
    all_input_ids = []
    all_labels = []

    for i in range(len(examples["question"])):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"问题：{examples['question'][i]}"},
            {"role": "assistant", "content": examples["query"][i]},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 找到 assistant 回复的起始位置，只对回复部分计算 loss
        assistant_start = text.rfind(examples["query"][i])
        if assistant_start == -1:
            continue

        full_ids = tokenizer(text, truncation=True, max_length=MAX_LENGTH)
        prompt_len = len(tokenizer(text[:assistant_start], truncation=True,
                                   max_length=MAX_LENGTH)["input_ids"])

        labels = full_ids["input_ids"].copy()
        labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

        all_input_ids.append(full_ids["input_ids"])
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


def main():
    print(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"基座模型: {BASE_MODEL}")
    print(f"训练数据: {DATA_FILE}")
    print(f"输出路径: {OUTPUT_DIR}")

    # 加载数据
    train_ds, val_ds = load_data()

    # 加载模型和 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # 挂 LoRA（r=4 减少 CPU 训练负担）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenize
    def tokenize_fn(examples):
        return tokenize(examples, tokenizer)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    # 训练
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        fp16=False,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    print("\n开始训练...")
    trainer.train()

    # 保存最终 LoRA 权重
    final_dir = OUTPUT_DIR / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nLoRA 适配器已保存到: {final_dir}")

    # 验证：测试一条
    print("\n===== 验证 =====")
    from peft import PeftModel
    test = "公司破产了股东要承担什么法律责任"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"问题：{test}"},
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=64, temperature=0.1)
    result = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"输入: {test}")
    print(f"输出: {result.strip()}")


if __name__ == "__main__":
    main()
