import math

import torch
import torch.optim as optim


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                t += 1
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * g**2
                lr_t = lr * (1 - beta2**t) ** 0.5 / (1 - beta1**t)
                p.data -= lr_t * m / (v**0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] = t
                state["m"] = m
                state["v"] = v

        return loss


def get_cosine_lr(t: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int) -> float:
    if t < warmup_steps:
        return t / warmup_steps * lr_max
    elif warmup_steps <= t < cosine_steps:
        cos_lr = lr_min + 0.5 * (1 + math.cos((t - warmup_steps) / (cosine_steps - warmup_steps) * math.pi)) * (
            lr_max - lr_min
        )
        return cos_lr
    else:
        return lr_min
