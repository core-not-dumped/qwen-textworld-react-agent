import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 기존 linear weight freeze
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        # LoRA 파라미터 초기화
        self.A = nn.Parameter(torch.zeros((r, linear.in_features)))
        self.B = nn.Parameter(torch.zeros((linear.out_features, r)))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.lora_enabled = True

    def forward(self, x):
        if self.lora_enabled:   return self.linear(x) + (x @ self.A.T @ self.B.T) * self.scaling
        else:                   return self.linear(x)

    def enable_lora(self):
        self.lora_enabled = True

    def disable_lora(self):
        self.lora_enabled = False


def merge_lora(linear_lora: LoRALinear):
    """LoRA 적용된 Linear weight를 합쳐서 일반 Linear로 반환"""
    # LoRA 합산
    merged_weight = linear_lora.linear.weight + (linear_lora.B @ linear_lora.A) * linear_lora.scaling

    # 새로운 Linear layer 생성
    new_linear = nn.Linear(linear_lora.linear.in_features, linear_lora.linear.out_features, bias=linear_lora.linear.bias is not None)
    new_linear.weight.data = merged_weight.data.clone()
    if linear_lora.linear.bias is not None:
        new_linear.bias.data = linear_lora.linear.bias.data.clone()

    return new_linear

def enable_lora(model):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.enable_lora()

def disable_lora(model):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.disable_lora()

def apply_lora_for_casulLM(model, r, alpha, device):
    for layer in model.model.layers:
        # 기존 linear layer 저장
        old_q = layer.self_attn.q_proj
        old_k = layer.self_attn.k_proj
        old_v = layer.self_attn.v_proj
        old_o = layer.self_attn.o_proj

        # LoRA로 교체
        layer.self_attn.q_proj = LoRALinear(old_q, r=r, alpha=alpha).to(device)
        layer.self_attn.k_proj = LoRALinear(old_k, r=r, alpha=alpha).to(device)
        layer.self_attn.v_proj = LoRALinear(old_v, r=r, alpha=alpha).to(device)
        layer.self_attn.o_proj = LoRALinear(old_o, r=r, alpha=alpha).to(device)

def merge_lora_in_self_attn(model):
    for layer in model.model.layers:
        layer.self_attn.q_proj = merge_lora(layer.self_attn.q_proj)
        layer.self_attn.k_proj = merge_lora(layer.self_attn.k_proj)
        layer.self_attn.v_proj = merge_lora(layer.self_attn.v_proj)
        layer.self_attn.o_proj = merge_lora(layer.self_attn.o_proj)
