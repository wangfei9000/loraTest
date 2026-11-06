"""
Intel Mac CPU 友好的房地产估价 LoRA 微调脚本。
基座：Qwen/Qwen2.5-0.5B-Instruct（约 1GB，16GB 内存可跑）
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "housing_prices.jsonl"
OUT_DIR = ROOT / "outputs" / "housing-lora"
_LOCAL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
MODEL_ID = os.environ.get(
    "MODEL_ID",
    str(_LOCAL) if (_LOCAL / "config.json").exists() else "Qwen/Qwen2.5-0.5B-Instruct",
)

# Intel Mac：无 MPS/CUDA，强制 CPU + fp32
# 样本很少时提高 epoch，故意在训练集上过拟合，便于看到 LoRA 效果
DEVICE = "cpu"
MAX_LEN = 384
EPOCHS = 12
LR = 2e-4
BATCH = 1
GRAD_ACCUM = 4
LORA_R = 16
LORA_ALPHA = 32


def load_messages(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    assert DATA_PATH.exists(), f"缺少数据文件: {DATA_PATH}"

    print(f"加载基座: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE)
    # 关闭 cache，训练更稳
    model.config.use_cache = False

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    raw = load_messages(DATA_PATH)

    def tokenize_row(example: dict) -> dict:
        messages = example["messages"]
        # 完整对话（含 assistant 答案）
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # 仅到 user，用于把 prompt 段 labels 置为 -100（只学回答）
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer(full_text, truncation=True, max_length=MAX_LEN)
        prompt_len = len(
            tokenizer(prompt_text, truncation=True, max_length=MAX_LEN)["input_ids"]
        )
        prompt_len = min(prompt_len, len(full["input_ids"]))
        labels = full["input_ids"][:]
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    tokenized = Dataset.from_list(raw).map(
        tokenize_row, remove_columns=["messages"]
    )

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_strategy="epoch",
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=True,
        label_names=["labels"],
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(
        f"开始训练（{len(raw)} 条 × {EPOCHS} epoch，Intel CPU，"
        "预计十几～几十分钟）…"
    )
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"LoRA 已保存到: {OUT_DIR}")
    print("建议用训练集里出现过的房源再测，例如默认望京 / 张江样例。")


if __name__ == "__main__":
    main()
