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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "housing_prices.jsonl"
OUT_DIR = ROOT / "outputs" / "housing-lora"
# 优先用本地模型（见 download_model.sh）；也可设环境变量 MODEL_ID
_LOCAL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
MODEL_ID = os.environ.get(
    "MODEL_ID",
    str(_LOCAL) if (_LOCAL / "config.json").exists() else "Qwen/Qwen2.5-0.5B-Instruct",
)

# Intel Mac：无 MPS/CUDA，强制 CPU + fp32
DEVICE = "cpu"
MAX_LEN = 384
EPOCHS = 3
LR = 1e-4
BATCH = 1
GRAD_ACCUM = 4
LORA_R = 8
LORA_ALPHA = 16


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

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # 少挂几层，CPU 更轻
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    raw = load_messages(DATA_PATH)

    def to_text(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = Dataset.from_list(raw).map(to_text)

    def tokenize(batch: dict) -> dict:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LEN,
            padding=False,
        )
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

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
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("开始训练（Intel Mac CPU，预计十几分钟到一小时，取决于机器）…")
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"LoRA 已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
