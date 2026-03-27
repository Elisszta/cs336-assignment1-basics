from typing import cast

import torch
from torch import Tensor

# from torch.optim.lr_scheduler import LRScheduler
import torch.nn as nn
import einx
import math
from jaxtyping import Float, Int

from collections.abc import Callable, Iterable


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
    ) -> Float[Tensor, ""]:
        max_logits = einx.max("b v -> b", inputs).values
        correct_tok = einx.get_at("b [v], b -> b", inputs, targets)
        exp_shifted = cast(Tensor, einx.subtract("b v, b -> b v", inputs, max_logits)).exp()
        log_sum = einx.sum("b v -> b", exp_shifted).log()
        loss = cast(Tensor, log_sum) + cast(Tensor, max_logits) - correct_tok
        return einx.mean("b -> ", loss)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Learning rate shouldn't under 0.")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, Closure: Callable | None = None):
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
    def step(self, Closure: Callable | None = None):
        loss = None if Closure is None else Closure()
        for group in self.param_groups:
            lr, b1, b2, eps, decay = group["lr"], group["b1"], group["b2"], group["eps"], group["decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if decay != 0:
                    p.mul_(1 - lr * decay)  # params_t = params_t-1 * (1 - lr * decay)
                state = self.state[p]
                t, m, v = (
                    state.setdefault("t", 0),
                    state.setdefault("m", torch.zeros_like(p)),
                    state.setdefault("v", torch.zeros_like(p)),
                )
                t += 1
                g = p.grad
                m.mul_(b1).add_(g, alpha=(1 - b1))  # m_t = m_t-1 * b1 + g * (1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)  # v_t = v_t-1 * b2 + g^2 * (1 - b2)
                lrt = lr * math.sqrt(1 - b2**t) / (1 - b1**t)  # lr_t = lr * sqrt(1 - b2^t) / (1 - b1^t)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lrt)  # params_t = params_t-1 - lrt * (m_t / (sqrt(v_t) + eps))
                state["t"], state["m"], state["v"] = t, m, v
        return loss


class CosineAnnealingLR:
    def __init__(
        self,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):
        self.max_lr, self.min_lr, self.warmup_iters, self.cosine_iters = (
            max_learning_rate,
            min_learning_rate,
            warmup_iters,
            cosine_cycle_iters,
        )

    def get_lr(self, it: int) -> float:
        if it < self.warmup_iters:
            return it / self.warmup_iters * self.max_lr
        elif it > self.cosine_iters:
            return self.min_lr
        else:
            return self.min_lr + 0.5 * (
                1 + math.cos((it - self.warmup_iters) * math.pi / (self.cosine_iters - self.warmup_iters))
            ) * (self.max_lr - self.min_lr)

    def __call__(self, it: int):
        return self.get_lr(it)


class GradientClipping:
    def __init__(self, max_l2_norm: float, eps=1e-6):
        self.max_l2, self.eps = max_l2_norm, eps

    @torch.no_grad()
    def clipping(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        total_l2_norm = 0.0  # L2 Norm for the whole parameters
        grads = [p.grad for p in parameters if p.grad is not None]
        for g in grads:
            total_l2_norm += torch.linalg.vector_norm(g, ord=2) ** 2
        total_l2_norm = math.sqrt(total_l2_norm)
        if total_l2_norm > self.max_l2:
            coef = self.max_l2 / (total_l2_norm + self.eps)
            for g in grads:
                g.mul_(coef)

    def __call__(self, parameters: Iterable[torch.nn.Parameter]):
        return self.clipping(parameters)
