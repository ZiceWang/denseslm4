import torch
from torch import nn
from torch.nn import functional as F
from denseslm4.layers.deltaformer import DeltaFormerAttention
from denseslm4.layers.mla import MultiheadLatentAttention
from denseslm4.layers.latent_deltaformer import LatentDeltaFormerAttention
from denseslm4.layers.mamba2 import Mamba2
from denseslm4.configuration_denseslm4 import DenseSLM4Config
from transformers.activations import ACT2FN

class DenseMLP(nn.Module):
    def __init__(self, config:DenseSLM4Config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)
    
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
        
        self.mlp = DenseMLP(config)

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


class IdentityHC(nn.Module):
    """
    Identity Hyper-Connection (IHC) following arxiv:2409.19606.
    H_res = I (identity matrix).
    hidden_size must be divisible by num_streams.
    """
    def __init__(self, hidden_size: int, num_streams: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_streams = num_streams

        assert hidden_size % num_streams == 0, (
            f"IdentityHC error: hidden_size({hidden_size}) must be divisible by num_streams({num_streams})"
        )
        self.stream_dim = hidden_size // num_streams

        self.norm = nn.LayerNorm(hidden_size)
        self.h_pre = nn.Linear(hidden_size, num_streams, bias=False)
        self.h_post = nn.Linear(hidden_size, num_streams, bias=False)

        nn.init.zeros_(self.h_pre.weight)
        nn.init.ones_(self.h_post.weight)

    def forward(self, x: torch.Tensor, fn) -> torch.Tensor:
        """
        x: input features [B, L, hidden_size]
        fn: branch function (Attn or MLP)
        returns: IHC fused features
        """
        B, L, D = x.shape
        n = self.num_streams
        C = self.stream_dim

        x_streams = x.view(B, L, n, C)

        x_norm = self.norm(x)
        pre = torch.tanh(self.h_pre(x_norm)).unsqueeze(-1)
        post = torch.tanh(self.h_post(x_norm)).unsqueeze(-1)

        z = fn(x)
        z = z.view(B, L, n, C)

        out = x_streams + post * z
        return out.flatten(-2)

class DenseMLAIHCBlock(nn.Module):
    """
    MLA block with IHC for both Attention and MLP branches.
    Follows arxiv:2409.19606 with H_res=I (identity matrix).
    """
    def __init__(self, config: DenseSLM4Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_streams = config.ihc_num_streams

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiheadLatentAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            q_lora_rank=config.mla_q_lora_rank,
            kv_lora_rank=config.mla_kv_lora_rank,
            qk_nope_head_dim=config.mla_qk_nope_head_dim,
            qk_rope_head_dim=config.mla_qk_rope_head_dim,
            v_head_dim=config.mla_v_head_dim,
            qk_head_dim=None,
            layer_idx=layer_idx,
        )

        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = DenseMLP(config)

        self.ihc_attn = IdentityHC(self.hidden_size, self.num_streams)
        self.ihc_mlp = IdentityHC(self.hidden_size, self.num_streams)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        def attn_fn(x):
            x = self.input_layernorm(x)
            x, _, _ = self.self_attn(x, attention_mask=None)
            return x
        hidden_states = self.ihc_attn(hidden_states, attn_fn)

        def mlp_fn(x):
            x = self.post_attention_layernorm(x)
            return self.mlp(x)
        hidden_states = self.ihc_mlp(hidden_states, mlp_fn)

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
