#!/usr/bin/env bash
# 从 hf-mirror 下载 Qwen2.5-0.5B-Instruct 到本地（国内网络友好）
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)/models/Qwen2.5-0.5B-Instruct"
mkdir -p "$DIR"
cd "$DIR"
BASE="https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors; do
  echo "==> $f"
  curl -L --fail -C - -o "$f" "$BASE/$f"
done
echo "完成: $DIR"
ls -lh
