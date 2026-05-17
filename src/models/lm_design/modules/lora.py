import re
import math
import torch
import torch.nn as nn


def _get_submodules(model, key):
    """
    Reference: (1) https://github.com/huggingface/peft/blob/e6cd24c907565040ee1766a5735afe3d13a71164/src/peft/tuners/lora/model.py#L65
    """
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def decorate_lora(model, target_modules, lora_r, lora_alpha):
    for key, _ in model.named_modules():
        if isinstance(target_modules, str):
            target_module_found = re.fullmatch(target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in target_modules)

        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                new_module = LinearWithLoRA(target, in_features, out_features, lora_r, lora_alpha)
                setattr(parent, target_name, new_module)


class LinearWithLoRA(nn.Module):
    """
    Reference: (1) https://lightning.ai/lightning-ai/studios/code-lora-from-scratch
    """
    def __init__(self, linear, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.linear = linear
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.linear(x) + self.alpha * (x @ self.A @ self.B)


# todo: embedding with lora