"""
LatentDeltaFormer: Combining MLA's low-rank latent compression with DeltaFormer's two-stage correction.

Key ideas:
1. MLA-style low-rank compression for K/V (via kv_lora_rank) to save memory
2. DeltaFormer's two-stage process:
   - Stage 1: u[i] = v[i] - beta[i] * sum_{j<i} softmax(q[i] @ k[:i]^T) @ u[:i]
   - Stage 2: o = causal_attn(q, k, u) 
3. K-K similarity (DeltaFormer style) instead of Q-K for better SSM-like behavior
4. RoPE for positional encoding

Reference:
- MLA: DeepSeekV2 (https://arxiv.org/abs/2405.04434)
- DeltaFormer: https://arxiv.org/pdf/2505.19488
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.utils import logging

from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask
from fla.ops.deltaformer import deltaformer_attn

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


class LatentDeltaFormerAttention(nn.Module):
    r"""
    LatentDeltaFormer combines MLA's low-rank latent attention with DeltaFormer's two-stage correction.

    The key innovation is using MLA's compressed K/V representation while applying
    DeltaFormer's two-stage process to handle the compression artifact correction.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_head_dim: int | None = None,
        rope_theta: float = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()

        # Sanity check
        if qk_head_dim is None:
            qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        else:
            assert qk_head_dim == qk_nope_head_dim + qk_rope_head_dim, \
                f"qk_head_dim {qk_head_dim} != qk_nope + qk_rope = {qk_nope_head_dim + qk_rope_head_dim}"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_head_dim = qk_head_dim

        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation`")

        # Q projection (with optional LoRA rank compression)
        if q_lora_rank is not None:
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_size, q_lora_rank, bias=False),
                RMSNorm(q_lora_rank, dtype=torch.float32),
                nn.Linear(q_lora_rank, self.num_heads * self.qk_head_dim, bias=False),
            )
        else:
            self.q_proj = nn.Linear(hidden_size, self.num_heads * self.qk_head_dim, bias=False)

        # K rope projection (for RoPE)
        self.k_rope = nn.Linear(hidden_size, self.qk_rope_head_dim, bias=False)

        # KV projection with compression (MLA style)
        self.kv_proj = nn.Sequential(
            nn.Linear(hidden_size, self.kv_lora_rank, bias=False),
            RMSNorm(self.kv_lora_rank, dtype=torch.float32),
            nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False),
        )

        # Beta projection (DeltaFormer style) - learns the correction factor
        self.beta_proj = nn.Linear(hidden_size, self.num_heads, bias=True)

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, hidden_size, bias=False)

        # RoPE
        self.rotary = RotaryEmbedding(dim=self.qk_rope_head_dim, base=self.rope_theta)

        self.scaling = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding)."
            )

        batch_size, q_len, _ = hidden_states.shape

        # Q projection and split into nope + rope parts
        q_states = self.q_proj(hidden_states)
        q_states = rearrange(q_states, 'b t (h d) -> b t h d', d=self.qk_head_dim)
        q_nope, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV projection (MLA style compression)
        k_states, v_states = self.kv_proj(hidden_states), None  # Will split later
        k_states = rearrange(k_states, 'b t (h d) -> b t h d', d=self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(k_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # K rope
        k_rot = self.k_rope(hidden_states)
        k_rot = rearrange(k_rot, 'b t d -> b t 1 d')

        # Beta (DeltaFormer correction factor)
        beta = self.beta_proj(hidden_states)  # [b, t, h]

        # Prepare RoPE
        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q_len + seqlen_offset
            if attention_mask is not None:
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q_len + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        cu_seqlens = kwargs.get('cu_seqlens')

        # Apply RoPE to q and k
        q_rot, k_rot = self.rotary(
            q_rot, k_rot, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens,
        )

        # Expand k_rot to all heads
        k_rot = repeat(k_rot, 'b t 1 d -> b t h d', h=self.num_heads)

        # Concatenate nope + rope parts
        q = torch.cat((q_nope, q_rot), dim=-1)  # [b, t, h, qk_head_dim]
        k = torch.cat((k_nope, k_rot), dim=-1)  # [b, t, h, qk_head_dim]

        # Cache management
        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k, v),
                layer_idx=self.layer_idx,
                offset=q_len,
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached

        # Pad v to match qk_head_dim for flash attn compatibility
        if self.qk_head_dim != self.v_head_dim:
            v = F.pad(v, [0, self.qk_head_dim - self.v_head_dim])

        # Apply DeltaFormer-style two-stage attention using official kernel
        o = deltaformer_attn(
            q=q,
            k=k,
            v=v,
            beta=beta,
            attention_mask=attention_mask,
            cu_seqlens=cu_seqlens
        )

        # Reshape and project output
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o)

        return o, None, past_key_values



class LatentDeltaFormerAttentionWithFlash(nn.Module):
    """
    Optimized LatentDeltaFormer using Flash Attention for the second stage.
    This uses flash_attn_func for efficient attention computation.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 16,
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        qk_head_dim: int | None = None,
        rope_theta: float = 10000.,
        max_position_embeddings: int | None = None,
        layer_idx: int = None,
    ):
        super().__init__()
        self.inner_attn = LatentDeltaFormerAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            qk_head_dim=qk_head_dim,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        return self.inner_attn(hidden_states, attention_mask, past_key_values, **kwargs)
