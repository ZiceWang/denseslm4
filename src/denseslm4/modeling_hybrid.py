import torch
from torch import nn
from torch.nn import functional as F
from fla.layers.deltaformer import DeltaFormerAttention
from fla.layers.mla import MultiheadLatentAttention
from denseslm4.latent_deltaformer import LatentDeltaFormerAttention
from fla.layers.mamba2 import Mamba2
from denseslm4.configuration_denseslm4 import DenseSLM4Config

class DenseMLABlock(nn.Module):
    """A standard Transformer block holding the MLA Attention and an MLP."""
    def __init__(self, config:DenseSLM4Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        
        # Use fla MLA with RoPE
        # qk_head_dim will be auto-computed as qk_nope + qk_rope when None
        mla_q_lora_rank = config.mla_q_lora_rank
        mla_kv_lora_rank = config.mla_kv_lora_rank
        mla_qk_nope_head_dim = config.mla_qk_nope_head_dim
        mla_qk_rope_head_dim = config.mla_qk_rope_head_dim
        mla_v_head_dim = config.mla_v_head_dim
        
        self.self_attn = MultiheadLatentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=mla_q_lora_rank,
            kv_lora_rank=mla_kv_lora_rank,
            qk_nope_head_dim=mla_qk_nope_head_dim,
            qk_rope_head_dim=mla_qk_rope_head_dim,
            v_head_dim=mla_v_head_dim,
            qk_head_dim=None,  # Let it compute automatically from qk_nope + qk_rope
            layer_idx=layer_idx,
        )
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # fla MLA expects (hidden_states, attention_mask, past_key_values, ...)
        hidden_states, _, _ = self.self_attn(hidden_states, attention_mask=None)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class DenseDeltaFormerMixerWrapper(nn.Module):
    """Wraps DeltaFormer from fla and isolates its distinct configuration needs."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.mixer = DeltaFormerAttention(
            hidden_size=config.hidden_size,
            num_heads=config.mamba_num_heads,
            num_kv_heads=config.mamba_num_heads,  # Use same for simplicity
            qkv_bias=False,
            qk_norm=False,
            rope_theta=10000.0,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # DeltaFormer returns (output, attentions, past_key_values)
        output, _, _ = self.mixer(hidden_states, attention_mask=None)
        return output


class DenseDeltaFormerBlock(nn.Module):
    """A DeltaFormer block consisting of a Norm and the DeltaFormer Mixer. No MLP."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size)
        self.mixer = DenseDeltaFormerMixerWrapper(config, layer_idx)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states)
        return residual + hidden_states


class DenseMamba2Block(nn.Module):
    """A Mamba2 block using fla's Mamba2, matching Mamba2Block structure."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        # Mamba2 requires num_heads * head_dim == expand * hidden_size
        # With hidden_size=512, expand=2 → need num_heads * head_dim = 1024
        # Use head_dim=64, num_heads=16 to satisfy this
        head_dim = 64
        num_heads = config.hidden_size * 2 // head_dim  # 1024 / 64 = 16
        self.self_attn = Mamba2(
            num_heads=num_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            state_size=config.ldf_q_lora_rank,
            expand=2,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states)[0]
        return residual + hidden_states


class DenseLatentDeltaFormerBlock(nn.Module):
    """A LatentDeltaFormer block: combines MLA's low-rank compression with DeltaFormer's correction."""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.norm = nn.RMSNorm(config.hidden_size)
        self.self_attn = LatentDeltaFormerAttention(
            hidden_size=config.hidden_size,
            num_heads=config.ldf_num_heads,
            q_lora_rank=config.ldf_q_lora_rank,
            kv_lora_rank=config.ldf_kv_lora_rank,
            qk_nope_head_dim=config.ldf_qk_nope_head_dim,
            qk_rope_head_dim=config.ldf_qk_rope_head_dim,
            v_head_dim=config.ldf_v_head_dim,
            rope_theta=10000.0,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states, _, _ = self.self_attn(hidden_states, attention_mask=None)
        return residual + hidden_states
