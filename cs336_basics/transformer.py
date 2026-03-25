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


# class chunkedRotaryPositionalEmbedding(nn.Module):
#     def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
#         super().__init__()

#         # Basic rotation freq
#         k_indices = torch.arange(0, d_k, 2, device=device).float()  # (2k - 2)
#         rot_freq = 1.0 / (theta ** (k_indices / d_k))  # theta^((2k - 2) / d)

#         # Pre-calculate rotation angle for max_seq_len
#         t = torch.arange(max_seq_len, device=device).float()
#         angles = cast(Tensor, einx.dot("seq_len, d2 -> seq_len d2", t, rot_freq))

#         # Register into buffer
#         self.register_buffer("cos_emb", angles.cos())  # max_seq_len x d/2
#         self.register_buffer("sin_emb", angles.sin())

#     def forward(
#         self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]
#     ) -> Float[Tensor, "... seq_len d_k"]:
#         x1, x2 = einx.rearrange("... (two d2) -> two ... d2", x, two=2)  # ... x seqlen x d/2
#         cos_idx, sin_idx = self.cos_emb[token_positions], self.sin_emb[token_positions]

#         while cos_idx.ndim < x1.ndim:
#             cos_idx = cos_idx.unsqueeze(-3)
#             sin_idx = sin_idx.unsqueeze(-3)

#         x1_new = x1 * cos_idx - x2 * sin_idx
#         x2_new = x2 * cos_idx + x1 * sin_idx
#         return torch.cat([x1_new, x2_new], -1)


class RotaryPositionalEmbedding(nn.Module):
    cos_emb: Tensor
    sin_emb: Tensor

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
        return cast(Tensor, einx.rearrange("two ... d2 -> ... (d2 two)", combine, two=2))


class Softmax(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
        max_val = torch.max(in_features, dim=self.dim, keepdim=True).values
        exp_x = torch.exp(in_features - max_val)
        sum_val = torch.sum(exp_x, dim=self.dim, keepdim=True)
        return exp_x / sum_val


class BasicAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax_layer = Softmax(-1)

    def forward(
        self,
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        sqrt_dim = math.sqrt(Q.shape[-1])
        scores = einx.dot("... queries d_k, ... keys d_k -> ... queries keys", Q, K) / sqrt_dim
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        sm_scores = self.softmax_layer(scores)
        return einx.dot("... queries keys, ... keys d_v -> ... queries d_v", sm_scores, V)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float | None = None,
        seq_max_len: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.attention = BasicAttention()
        self.head, self.use_rope, self.device = num_heads, use_rope, device
        if self.use_rope:
            self.RoPE = RotaryPositionalEmbedding(
                cast(float, theta), int(d_model / num_heads), cast(int, seq_max_len), device
            )

    def forward(
        self,
        q_proj_weight: Float[Tensor, " d_k d_in"],
        k_proj_weight: Float[Tensor, " d_k d_in"],
        v_proj_weight: Float[Tensor, " d_v d_in"],
        o_proj_weight: Float[Tensor, " d_model d_v"],
        in_features: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        Q, K, V = (
            einx.dot(
                "... seq_len d_in, (head d_h) d_in -> ... head seq_len d_h", in_features, q_proj_weight, head=self.head
            ),
            einx.dot(
                "... seq_len d_in, (head d_h) d_in -> ... head seq_len d_h", in_features, k_proj_weight, head=self.head
            ),
            einx.dot(
                "... seq_len d_in, (head d_h) d_in -> ... head seq_len d_h", in_features, v_proj_weight, head=self.head
            ),
        )
        if token_positions is None:
            token_positions = torch.arange(in_features.shape[-2], device=in_features.device)
        if self.use_rope:
            Q, K = self.RoPE(Q, token_positions), self.RoPE(K, token_positions)
        mask = torch.tril(torch.ones(in_features.shape[-2], in_features.shape[-2], device=in_features.device).bool())
        attention_val = einx.rearrange("... head seq_len d_h -> ... seq_len (head d_h)", self.attention(Q, K, V, mask))
        return einx.dot(
            "... seq_len d_v, d_model d_v -> ... seq_len d_model", cast(Tensor, attention_val), o_proj_weight
        )


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, num_heads, True, theta, max_seq_len)
        self.FFN = SwiGLU(d_model, d_ff)
        self.PreNormAttn, self.PreNormFFN = RMSNorm(d_model), RMSNorm(d_model)

    def forward(
        self,
        weights: dict[str, Tensor],
        in_features: Float[Tensor, " batch sequence_length d_model"],
    ):
        self.PreNormAttn.weight.data, self.PreNormFFN.weight.data = weights["ln1.weight"], weights["ln2.weight"]
        self.FFN.weight_1.data, self.FFN.weight_2.data, self.FFN.weight_3.data = (
            weights["ffn.w1.weight"],
            weights["ffn.w2.weight"],
            weights["ffn.w3.weight"],
        )
        in_features += self.MHA(
            weights["attn.q_proj.weight"],
            weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],
            weights["attn.output_proj.weight"],
            self.PreNormAttn(in_features),
        )
        in_features += self.FFN(self.PreNormFFN(in_features))
        return in_features


class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.input_emb_layer = Embedding(vocab_size, d_model)
        self.transformer_layer = Transformer(d_model, num_heads, d_ff, context_length, rope_theta)
        self.post_norm_layer = RMSNorm(d_model)
        self.output_emb_layer = Linear(d_model, vocab_size)
        self.num_layers = num_layers
        # self.softmax_layer = Softmax(-1)

    def forward(
        self,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
    ):

        self.input_emb_layer.weight.data, self.post_norm_layer.weight.data, self.output_emb_layer.weight.data = (
            weights["token_embeddings.weight"],
            weights["ln_final.weight"],
            weights["lm_head.weight"],
        )
        data = self.input_emb_layer(in_indices)
        for num_layer in range(self.num_layers):
            transformer_weights = {
                "ln1.weight": weights[f"layers.{num_layer}.ln1.weight"],
                "ln2.weight": weights[f"layers.{num_layer}.ln2.weight"],
                "ffn.w1.weight": weights[f"layers.{num_layer}.ffn.w1.weight"],
                "ffn.w2.weight": weights[f"layers.{num_layer}.ffn.w2.weight"],
                "ffn.w3.weight": weights[f"layers.{num_layer}.ffn.w3.weight"],
                "attn.q_proj.weight": weights[f"layers.{num_layer}.attn.q_proj.weight"],
                "attn.k_proj.weight": weights[f"layers.{num_layer}.attn.k_proj.weight"],
                "attn.v_proj.weight": weights[f"layers.{num_layer}.attn.v_proj.weight"],
                "attn.output_proj.weight": weights[f"layers.{num_layer}.attn.output_proj.weight"],
            }
            data = self.transformer_layer(transformer_weights, data)
        data = self.post_norm_layer(data)
        data = self.output_emb_layer(data)
        # data = self.softmax_layer(data)
        return data
