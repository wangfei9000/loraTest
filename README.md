# 房地产估价 LoRA 微调 Demo

在 **Intel Mac（CPU、无独显）** 上，用小模型 `Qwen2.5-0.5B-Instruct` + LoRA，微调一个「根据房源信息给出估价」的对话模型。

> 这是学习 / 流程演示项目。当前样本少、模型小，**输出价格不能当真**。生产级估价更适合表格特征 + XGBoost/LightGBM；LLM 更适合学报价话术与输出格式。

## 环境要求

- macOS（本项目按 Intel i5 + 16GB 内存调过）
- Python 3.10+
- 磁盘约 2GB（基座模型 ~1GB + 依赖）

## 目录结构

```
loraTest/
├── data/housing_prices.jsonl   # 训练数据（对话格式）
├── models/Qwen2.5-0.5B-Instruct # 本地基座（需先下载）
├── outputs/housing-lora/       # 训练后的 LoRA 权重
├── download_model.sh           # 从 hf-mirror 下载基座
├── train.py                    # 微调
├── infer.py                    # 推理
├── requirements.txt
├── test.py                     # LoRA 线性层玩具实现（仅示意，不参与训练）
└── README.md
```

## 快速开始

```bash
cd /Users/edy/Downloads/wangfei/loraTest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 下载基座（国内走 hf-mirror，约 1GB）
bash download_model.sh

# 训练（CPU 约数分钟～十几分钟）
python train.py

# 推理
python infer.py
python infer.py --query "城市：上海；区域：浦东新区张江；面积：105㎡；户型：3室2厅；楼层：低层/11层；建成：2018；装修：精装；近地铁：是"

# 对比：不加 LoRA，只看基座
python infer.py --base-only
```

可选：用环境变量指定模型路径：

```bash
export MODEL_ID=/path/to/your/model
python train.py
```

## 训练配置（默认）

| 项 | 值 |
|---|---|
| 基座 | Qwen2.5-0.5B-Instruct |
| 设备 | CPU + float32 |
| LoRA rank / alpha | 8 / 16 |
| 目标层 | `q_proj`, `v_proj` |
| 可训练参数占比 | ~0.11% |
| batch / 梯度累积 | 1 / 4 |
| 序列长度 | 384 |
| epoch / 学习率 | 3 / 1e-4 |

可在 `train.py` 顶部常量处修改。

## 数据格式

`data/housing_prices.jsonl`，每行一条 JSON，含 `messages`（system / user / assistant）：

```json
{
  "messages": [
    {"role": "system", "content": "你是房地产估价助手。..."},
    {"role": "user", "content": "城市：北京；区域：朝阳区望京；面积：89㎡；..."},
    {"role": "assistant", "content": "估价：单价约 85000 元/㎡，总价约 756 万元。\n依据：..."}
  ]
}
```

建议 user 字段统一包含：城市、区域、面积、户型、楼层、建成年代、装修、是否近地铁。  
效果要变好，请换成真实成交/挂牌数据， ideally 几百～几千条后再训。

## 常见问题

1. **Hugging Face 连不上**  
   用 `bash download_model.sh` 从镜像拉到本地，脚本会优先读 `models/Qwen2.5-0.5B-Instruct`。

2. **NumPy / torch 报错**  
   保持 `numpy<2`（已写在 `requirements.txt`）。

3. **内存不够 / 太慢**  
   在 `train.py` 里减小 `MAX_LEN`、`EPOCHS`，或只挂更少 LoRA 层。

4. **估价很离谱**  
   正常：当前只有几十条模拟数据。先加数据、再加 epoch，不要指望 0.5B 小模型做准房价回归。

## 说明

- `test.py` 是手写 `LoRALinear` 示意，**不能**用来微调现有大模型。
- 真正微调走 `peft`：冻结基座，只保存 adapter（`outputs/housing-lora/`）。
- 推理时加载：基座 + adapter（见 `infer.py`）。
