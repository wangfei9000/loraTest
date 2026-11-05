import torch
import torch.nn as nn

class LoRALinear(nn.Module):

    def __init__(self, in_features, out_features,
                 rank=8, alpha=16):

        super().__init__()

        self.linear = nn.Linear(
            in_features, out_features
        )

        # 冻结原始权重
        for p in self.linear.parameters():
            p.requires_grad = False

        # LoRA 参数
        self.A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )

        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

        self.scale = alpha / rank

    def forward(self, x):

        original = self.linear(x)

        lora = (x @ self.A.T) @ self.B.T

        return original + self.scale * lora
