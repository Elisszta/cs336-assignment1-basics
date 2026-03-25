from typing import cast

import torch
from torch import Tensor
import torch.nn as nn
import einx
import math
from jaxtyping import Float, Int, Bool


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


class chunkedRotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        # Basic rotation freq
        k_indices = torch.arange(0, d_k, 2, device=device).float()  # (2k - 2)
        rot_freq = 1.0 / (theta ** (k_indices / d_k))  # theta^((2k - 2) / d)

        # Pre-calculate rotation angle for max_seq_len
        t = torch.arange(max_seq_len, device=device).float()
        angles = cast(Tensor, einx.dot("seq_len, d2 -> seq_len d2", t, rot_freq))

        # Register into buffer
        self.register_buffer("cos_emb", angles.cos())  # max_seq_len x d/2
        self.register_buffer("sin_emb", angles.sin())

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        x1, x2 = einx.rearrange("... (two d2) -> two ... d2", x, two=2)  # ... x seqlen x d/2
        cos_idx, sin_idx = self.cos_emb[token_positions], self.sin_emb[token_positions]

        while cos_idx.ndim < x1.ndim:
            cos_idx = cos_idx.unsqueeze(-3)
            sin_idx = sin_idx.unsqueeze(-3)

        x1_new = x1 * cos_idx - x2 * sin_idx
        x2_new = x2 * cos_idx + x1 * sin_idx
        return torch.cat([x1_new, x2_new], -1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        super().__init__()

        # Basic rotation freq
        k_indices = torch.arange(0, d_k, 2, device=device).float()  # (2k - 2)
        rot_freq = 1.0 / (theta ** (k_indices / d_k))  # theta^((2k - 2) / d)

        # Pre-calculate rotation angle for max_seq_len
        t = torch.arange(max_seq_len, device=device).float()
        angles = cast(Tensor, einx.dot("seq_len, d2 -> seq_len d2", t, rot_freq))

        # Register into buffer
        self.register_buffer("cos_emb", angles.cos())  # max_seq_len x d/2
        self.register_buffer("sin_emb", angles.sin())

    def forward(
        self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        x1, x2 = einx.rearrange("... (d2 two) -> two ... d2", x, two=2)  # ... x seqlen x d/2
        cos_idx, sin_idx = self.cos_emb[token_positions], self.sin_emb[token_positions]  # ... x seq_len x d/2

        # matching x's dim (Cause you don't know the pos's prev dim before seqlen)
        while cos_idx.ndim < x1.ndim:
            cos_idx = cos_idx.unsqueeze(-3)
            sin_idx = sin_idx.unsqueeze(-3)

        x1_new = x1 * cos_idx - x2 * sin_idx
        x2_new = x2 * cos_idx + x1 * sin_idx
        combine = torch.stack([x1_new, x2_new])
        return einx.rearrange("two ... d2 -> ... (d2 two)", combine, two=2)


class Softmax(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        max_val = torch.max(in_features, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(in_features - max_val)
        sum_val = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_val


class Attention(nn.Module):
    def __init__(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> None:
        super().__init__()
        self.Q, self.K, self.V = Q, K, V
        self.sqrt_dim = math.sqrt(Q.shape[-1])

    def forward(
        self,
    ) -> Float[Tensor, " ... queries d_v"]:
        score = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", self.Q, self.K) / self.sqrt_dim
        masked_sm
        sm = Softmax(-1)
        sm_score = sm(score)
