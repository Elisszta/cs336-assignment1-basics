import torch
from torch import Tensor
import torch.nn as nn
import einx
import math
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, self.std)

    def forward(self, x: torch.Tensor) -> Float[Tensor, "... out_feature"]:
        return einx.dot("... in_feature, out_feature in_feature -> ... out_feature", x, self.weight)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.std = math.sqrt(2 / (num_embeddings + embedding_dim))
        nn.init.trunc_normal_(self.weight, 0, self.std)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, "... embedding_dim"]:
        return einx.get_at("[num_embeddings] embedding_dim, ... -> ... embedding_dim", self.weight, token_ids)


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        ms = einx.mean("... [d]", x**2)
        rms = torch.sqrt(ms + self.eps)
        return einx.divide("... dim, ... -> ... dim", x, rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(
        self, d_in: int, d_hidden: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.std = math.sqrt(2 / (d_in + d_hidden))
        self.weight_1 = nn.Parameter(torch.empty(d_hidden, d_in, device=device, dtype=dtype))
        self.weight_2 = nn.Parameter(torch.empty(d_in, d_hidden, device=device, dtype=dtype))
        self.weight_3 = nn.Parameter(torch.empty(d_hidden, d_in, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight_1, 0, self.std)
        nn.init.trunc_normal_(self.weight_2, 0, self.std)
        nn.init.trunc_normal_(self.weight_3, 0, self.std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_in"]:
        gatepath_val = einx.dot("... d_in, d_hidden d_in -> ... d_hidden", x, self.weight_1)
        pathway_val = einx.dot("... d_in, d_hidden d_in -> ... d_hidden", x, self.weight_3)
        sigmoid_gate = torch.sigmoid(gatepath_val)
        swish_val = einx.multiply("..., ... -> ...", gatepath_val, sigmoid_gate)
        swiglu_val = einx.multiply("..., ... -> ...", swish_val, pathway_val)
        return einx.dot("... d_hidden, d_in d_hidden -> ... d_in", swiglu_val, self.weight_2)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.rotate

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        k_indices = torch.arange(0, self.d_k, 2)  # (2k - 2)
        rot_freq = 1.0 / (self.theta ** (k_indices / self.d_k))  # theta^((2k - 2) / d)
        angles = einx.dot("... seq_len, seq_len")
        x_rot, y_rot = einx.rearrange("... (two d2) -> two ... d2", x, two=2)
        x_rot_cos, y_rot_cos = einx.multiply("... d_k/2, d_k/2 -> ... d_k/2", x_rot, angles.cos())
        return x
