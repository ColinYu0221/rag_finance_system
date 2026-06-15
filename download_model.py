import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = r"C:\Users\wangx\Desktop\rag_finance_system\rag_finance_system\models\bge-reranker-v2-m3"

print("开始加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

print("开始加载 model...")
model = AutoModelForSequenceClassification.from_pretrained(
    path,
    local_files_only=True
)

print("加载成功！")