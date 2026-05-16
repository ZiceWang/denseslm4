"""Configuration for DenseSLM4MoE models."""

from __future__ import annotations

from transformers import PretrainedConfig


class DenseSLM4MoeConfig(PretrainedConfig):
    """DenseSLM4 variant with DeepSeek/Nemotron-H style aux-loss-free routed MoE."""

    model_type = "denseslm4moe"

    def __init__(
        self,
        vocab_size: int = 128,
        hidden_size: int = 768,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 16,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 128,
        dropout: float = 0.0,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,

        # DeltaFormer/Mamba2 parameters
        mamba_expand: int = 2,
        mamba_num_heads: int = 4,
        mamba_n_groups: int = 1,
        mamba_state_size: int = 64,
        mamba_conv_kernel: int = 4,

        # DeepSeekV3 MLA parameters
        num_key_value_heads: int = 4,
        mla_q_lora_rank: int | None = 32,
        mla_kv_lora_rank: int = 512,
        mla_qk_nope_head_dim: int = 128,
        mla_qk_rope_head_dim: int = 64,
        mla_v_head_dim: int = 128,

        # LatentDeltaFormer (LDF) parameters - separate from MLA
        ldf_num_heads: int = 4,
        ldf_qk_nope_head_dim: int = 32,
        ldf_qk_rope_head_dim: int = 32,
        ldf_v_head_dim: int = 64,
        ldf_q_lora_rank: int | None = 64,
        ldf_kv_lora_rank: int = 128,
        ihc_num_streams: int = 4,  # for IdentityHC
        hidden_act: str = "silu",

        # Aux-loss-free MoE parameters
        n_routed_experts: int = 64,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 1024,
        moe_shared_expert_intermediate_size: int = 1024,
        n_group: int = 8,
        topk_group: int = 2,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        router_bias_update_rate: float = 1e-3,
        router_score_func: str = "sigmoid",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        initializer_range: float = 0.02,
        use_projected_embedding: bool = False,
        projected_embedding_path: str | None = None,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        
        self.mamba_expand = mamba_expand
        self.mamba_num_heads = mamba_num_heads
        self.mamba_n_groups = mamba_n_groups
        self.mamba_state_size = mamba_state_size
        self.mamba_conv_kernel = mamba_conv_kernel
        
        self.num_key_value_heads = num_key_value_heads
        self.mla_q_lora_rank = mla_q_lora_rank
        self.mla_kv_lora_rank = mla_kv_lora_rank
        self.mla_qk_nope_head_dim = mla_qk_nope_head_dim
        self.mla_qk_rope_head_dim = mla_qk_rope_head_dim
        self.mla_v_head_dim = mla_v_head_dim

        # IdentityHC parameters
        self.ihc_num_streams = ihc_num_streams  # hidden_size must be divisible by num_streams for IdentityHC
        self.hidden_act = hidden_act

        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_shared_expert_intermediate_size = moe_shared_expert_intermediate_size
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.router_bias_update_rate = router_bias_update_rate
        self.router_score_func = router_score_func
        # LatentDeltaFormer params
        self.ldf_num_heads = ldf_num_heads
        self.ldf_qk_nope_head_dim = ldf_qk_nope_head_dim
        self.ldf_qk_rope_head_dim = ldf_qk_rope_head_dim
        self.ldf_v_head_dim = ldf_v_head_dim
        self.ldf_q_lora_rank = ldf_q_lora_rank
        self.ldf_kv_lora_rank = ldf_kv_lora_rank
        
        # Projected embedding params (for frozen embedding mode)
        self.use_projected_embedding = use_projected_embedding
        self.projected_embedding_path = projected_embedding_path
        
        self.initializer_range = initializer_range
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
