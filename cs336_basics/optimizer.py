import math

import torch
import torch.optim as optim


class AdamW(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
        # Save initial lr
        for group in params:
            group["initial_lr"] = group["lr"]

    def set_lr(self, lr):
        for group in self.param_groups:
            group["lr"] = lr

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

# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_linear_lr(it, warmup_iters, warmdown_ratio, num_iterations, final_lr_frac):
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac


def clip_grad_norm(params, max_norm: float = 1.0, eps: float = 1e-6):
    total_norm = sum([torch.sum(p.grad**2) for p in params if p.grad is not None])
    total_norm = total_norm**0.5
    if total_norm >= max_norm:
        with torch.no_grad():
            for p in params:
                if p.grad is not None:
                    p.grad.mul_(max_norm / (total_norm + eps))
