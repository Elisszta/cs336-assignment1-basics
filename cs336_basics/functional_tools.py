from typing import cast

import torch
from torch import Tensor
import torch.nn as nn
import einx
import math
from jaxtyping import Float, Int, Bool

from collections.abc import Callable, Iterable
from typing import Optional


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
    ) -> Float[Tensor, ""]:
        max_logits = einx.max("b v -> b", inputs).values
        correct_tok = inputs[torch.arange(inputs.shape[0], device=inputs.device), targets]
        exp_shifted = einx.subtract("b v, b -> b v", inputs, max_logits).exp()
        log_sum = einx.sum("b v -> b", exp_shifted).log()
        l = log_sum + max_logits - correct_tok
        return einx.mean("b -> ", l)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Learning rate shouldn't under 0.")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, Closure: Optional[Callable] = None):
        loss = None if Closure is None else Closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.setdefault("t", 0)
                grad = p.grad
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] += 1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0:
            raise ValueError("Learning rate shouldn't under 0.")
        defaults = {"lr": lr, "b1": betas[0], "b2": betas[1], "eps": eps, "decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, Closure: Optional[Callable] = None):
        loss = None if Closure is None else Closure()
        for group in self.param_groups:
            lr, b1, b2, eps, decay = group["lr"], group["b1"], group["b2"], group["eps"], group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if decay != 0:
                    p.mul_(1 - lr * decay)
                state = self.state[p]
                t, m, v = (
                    state.setdefault("t", 0),
                    state.setdefault("m", torch.zeros_like(p)),
                    state.setdefault("v", torch.zeros_like(p)),
                )
                t += 1
                g = p.grad
                m.mul_(b1).add_(g, alpha=(1 - b1))
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                lrt = lr * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lrt)
                state["t"], state["m"], state["v"] = t, m, v
        return loss
