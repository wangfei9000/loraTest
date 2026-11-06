# 房地产估价 LoRA 微调 Demo

在 **Intel Mac（CPU、无独显）** 上，用 `Qwen2.5-0.5B-Instruct` + LoRA，微调一个「根据房源信息给出估价」的对话模型。

> 这是学习 / 流程演示。当前 32 条模拟样本、0.5B 小模型，**输出价格不能当真**。生产级估价更适合表格特征 + XGBoost/LightGBM；LLM 更适合学报价话术和输出格式。

## 快速开始

- macOS（本项目按 Intel i5 + 16GB 内存调过）
- Python 3.10+
- 磁盘约 2GB（基座约 1GB + 依赖）

```bash
cd /Users/edy/Downloads/wangfei/loraTest
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 下载基座（国内走 hf-mirror，约 1GB）
bash download_model.sh

# 训练（CPU 约数分钟～十几分钟）
python train.py

# 推理：基座 + LoRA
python infer.py
python infer.py --query "城市：上海；区域：浦东新区张江；面积：105㎡；户型：3室2厅；楼层：低层/11层；建成：2018；装修：精装；近地铁：是"

# 对照：只看基座
python infer.py --base-only
```

换基座路径时：

```bash
export MODEL_ID=/path/to/your/model
python train.py
```

## 目录

```
loraTest/
├── data/housing_prices.jsonl    # 训练数据
├── models/Qwen2.5-0.5B-Instruct # 基座（download_model.sh 下载）
├── outputs/housing-lora/        # 训练产物（.gitignore，本地才有）
├── download_model.sh
├── train.py                     # 微调
├── infer.py                     # 推理
├── lizi.py                      # 手写 LoRALinear，只作示意，不参与训练
├── requirements.txt
└── README.md
```

真正微调走 `peft`：冻结基座，只训练并保存 adapter。`lizi.py` 不能用来微调现有大模型。

## 数据

`data/housing_prices.jsonl`，每行一条对话（system / user / assistant）：

```json
{
  "messages": [
    {"role": "system", "content": "你是房地产估价助手。..."},
    {"role": "user", "content": "城市：北京；区域：朝阳区望京；面积：89㎡；..."},
    {"role": "assistant", "content": "估价：单价约 85000 元/㎡，总价约 756 万元。\n依据：..."}
  ]
}
```

user 字段建议固定为：城市、区域、面积、户型、楼层、建成年代、装修、是否近地铁。  
要让估价像样，换成真实成交或挂牌数据，几百到几千条再训。

## 训练

默认配置在 `train.py` 顶部常量，直接改那里即可。

| 项 | 值 |
|---|---|
| 基座 | Qwen2.5-0.5B-Instruct，CPU + float32 |
| LoRA | rank 16，alpha 32，dropout 0，只挂 `q_proj` `k_proj` `v_proj` `o_proj` |
| batch / 梯度累积 | 1 / 4 |
| 序列长度 | 384 |
| epoch / 学习率 | 12 / 2e-4（样本少，故意在训练集上过拟合） |
| 损失 | 只对 assistant 回答段计算，prompt 段 label 为 -100 |
| 存盘 | 每个 epoch 一个 checkpoint，结束后再把最终 adapter 写到 `outputs/housing-lora/` |

## 训练产物

跑完 `python train.py` 后出现 `outputs/housing-lora/`（约 323 MB，已加入 `.gitignore`）。  
这次训练是 32 条 × 12 epoch，共 96 step，损失大约从 step 5 的 2.41 降到 step 95 的 0.29。

### 推理要带上的文件

`infer.py` 先加载基座 `models/Qwen2.5-0.5B-Instruct`，再用 `PeftModel.from_pretrained` 读 adapter 根目录。根目录里它实际只用两个文件：

| 文件 | 大小 | 作用 |
|---|---|---|
| `adapter_model.safetensors` | 8.3 MB | LoRA 权重 |
| `adapter_config.json` | 约 1 KB | rank、alpha、挂在哪些层 |

分词器从基座目录读取。只拷贝这两个文件到 `outputs/housing-lora/`，就可以跑 `python infer.py`。

### 同一目录里的其他文件

| 内容 | 用途 |
|---|---|
| `checkpoint-8` … `checkpoint-96`（各约 25 MB） | 每个 epoch 一份。其中 adapter 与根目录最终权重对应同一次训练的各步；`optimizer.pt`（约 16.7 MB）、`scheduler.pt`、`rng_state.pth` 用于断点续训 |
| `tokenizer.json`、`vocab.json`、`merges.txt` 等（约 15 MB） | 训练结束时顺手保存的分词器副本 |
| `README.md` | PEFT 自动生成的 model card 模板 |

只做推理时，删掉全部 `checkpoint-*` 大约能腾出 300 MB。

### 权重内容

基座 494,032,768 参数保持冻结，不写进 adapter。可训练的是 **2,162,688** 个 float32 参数（约 0.44%），共 192 个张量，挂在 24 层 `self_attn` 上。

张量名：

`base_model.model.model.layers.{0-23}.self_attn.{q,k,v,o}_proj.lora_{A,B}.weight`

| 模块 | `lora_A` | `lora_B` |
|---|---|---|
| `q_proj` | `[16, 896]` | `[896, 16]` |
| `k_proj` | `[16, 896]` | `[128, 16]` |
| `v_proj` | `[16, 896]` | `[128, 16]` |
| `o_proj` | `[16, 896]` | `[896, 16]` |

`hidden_size` 是 896。k/v 的 B 是 128，因为 GQA 只有 2 个 KV head，head 维度 64。  
前向在原线性层上加低秩增量：`ΔW = B @ A`，再乘 `lora_alpha / r`（32 / 16 = 2）。

## 常见问题

1. **Hugging Face 连不上**  
   用 `bash download_model.sh` 从镜像拉到本地。脚本和训练都会优先用 `models/Qwen2.5-0.5B-Instruct`。

2. **NumPy / torch 报错**  
   保持 `numpy<2`（已写在 `requirements.txt`）。

3. **内存不够或太慢**  
   在 `train.py` 里减小 `MAX_LEN`、`EPOCHS`，或减少 `target_modules`。

4. **估价很离谱**  
   当前只有几十条模拟数据。先加数据再加 epoch，不要指望 0.5B 小模型做准房价回归。
