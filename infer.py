"""加载微调后的 LoRA，对房源做估价推理。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parent
_LOCAL = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
BASE_MODEL = os.environ.get(
    "MODEL_ID",
    str(_LOCAL) if (_LOCAL / "config.json").exists() else "Qwen/Qwen2.5-0.5B-Instruct",
)
ADAPTER = ROOT / "outputs" / "housing-lora"

SYSTEM = (
    "你是房地产估价助手。根据房源信息给出合理单价（元/㎡）和总价（万元），"
    "并简要说明依据。只输出估价结论，不要闲聊。"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="城市：北京；区域：朝阳区望京；面积：89㎡；户型：2室1厅；"
        "楼层：中层/18层；建成：2015；装修：精装；近地铁：是",
    )
    parser.add_argument("--base-only", action="store_true", help="只用基座，不加 LoRA")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if not args.base_only:
        if not ADAPTER.exists():
            raise SystemExit(f"未找到 adapter: {ADAPTER}，请先运行 train.py")
        model = PeftModel.from_pretrained(model, str(ADAPTER))
        print(f"已加载 LoRA: {ADAPTER}")
    else:
        print("仅基座模型（未加载 LoRA）")

    model.eval()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.query},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    # 关掉 sampling 相关默认值，避免 temperature/top_p 告警
    gen_cfg = GenerationConfig(
        max_new_tokens=160,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)
    gen = out[0][inputs["input_ids"].shape[1] :]
    print("\n—— 估价结果 ——")
    print(tokenizer.decode(gen, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
