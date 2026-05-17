"""DenseSLM4MoE modeling code with aux-loss-free routed MoE blocks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutput

from .configuration_denseslm4moe import DenseSLM4MoeConfig
from .layers.mla import MultiheadLatentAttention
from .modules import DenseMamba2Block


class DenseSLM4MoeTopkRouter(nn.Module):
    """Aux-loss-free router with a non-trainable per-expert score correction bias."""

    def __init__(self, config: DenseSLM4MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))
        self.register_buffer("last_expert_load", torch.zeros(self.n_routed_experts), persistent=False)
        self.register_buffer("last_target_load", torch.tensor(0.0), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.reshape(-1, self.config.hidden_size)
        return F.linear(hidden_states.float(), self.weight.float())

    @torch.no_grad()
    def update_bias(self) -> None:
        update_rate = self.config.router_bias_update_rate
        if update_rate <= 0 or self.last_target_load <= 0:
            return
        load_error = torch.sign(self.last_target_load - self.last_expert_load)
        self.e_score_correction_bias.add_(update_rate * load_error)
        self.e_score_correction_bias.sub_(self.e_score_correction_bias.mean())


class DenseSLM4MoeExperts(nn.Module):
    """Routed expert collection: each expert is hidden -> moe_intermediate -> hidden."""

    def __init__(self, config: DenseSLM4MoeConfig) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        flat_expert_idx = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        token_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device).repeat_interleave(topk_indices.shape[1])

        sorted_expert_idx, order = torch.sort(flat_expert_idx)
        sorted_token_idx = token_idx.index_select(0, order)
        sorted_weights = flat_weights.index_select(0, order)
        tokens_per_expert = torch.bincount(sorted_expert_idx, minlength=self.num_experts)
        expert_hit = torch.nonzero(tokens_per_expert, as_tuple=False).flatten()

        start = 0
        for expert_idx in expert_hit.tolist():
            count = int(tokens_per_expert[expert_idx].item())
            end = start + count
            current_token_idx = sorted_token_idx[start:end]
            current_state = hidden_states.index_select(0, current_token_idx)
            current_hidden_states = F.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * sorted_weights[start:end, None]
            final_hidden_states.index_add_(0, current_token_idx, current_hidden_states.to(final_hidden_states.dtype))
            start = end

        return final_hidden_states.to(hidden_states.dtype)


class DenseSLM4MoeMLP(nn.Module):
    """Shared expert: hidden -> shared_intermediate -> hidden."""

    def __init__(self, config: DenseSLM4MoeConfig, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class DenseSLM4Moe(nn.Module):
    """DeepSeek/Nemotron-H style aux-loss-free MoE output: routed experts + shared expert."""

    def __init__(self, config: DenseSLM4MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = DenseSLM4MoeExperts(config)
        self.gate = DenseSLM4MoeTopkRouter(config)
        self.shared_experts = DenseSLM4MoeMLP(config, config.moe_shared_expert_intermediate_size)
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        if self.n_routed_experts % self.n_group != 0:
            raise ValueError("n_routed_experts must be divisible by n_group")
        if self.topk_group > self.n_group:
            raise ValueError("topk_group must be <= n_group")

    def route_tokens_to_experts(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.router_score_func == "sigmoid":
            router_scores = router_logits.sigmoid()
        elif self.config.router_score_func == "softmax":
            router_scores = router_logits.softmax(dim=-1)
        else:
            raise ValueError("router_score_func must be 'sigmoid' or 'softmax'")

        scores_for_choice = router_scores + self.gate.e_score_correction_bias.view(1, -1)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(min(2, self.n_routed_experts // self.n_group), dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        if self.training:
            with torch.no_grad():
                load = torch.bincount(topk_indices.reshape(-1), minlength=self.config.n_routed_experts).to(
                    dtype=torch.float32,
                    device=router_logits.device,
                )
                self.gate.last_expert_load.copy_(load)
                self.gate.last_target_load.copy_(load.mean())

        return topk_indices, topk_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        original_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        routed_states = self.experts(flat_hidden_states, topk_indices, topk_weights).view(*original_shape)
        if self.training:
            self.update_expert_bias()
        return routed_states + self.shared_experts(residual)

    @torch.no_grad()
    def update_expert_bias(self) -> None:
        self.gate.update_bias()


class DenseSLM4MoePreTrainedModel(PreTrainedModel):
    """Base class with Transformers-compatible initialization."""

    config_class = DenseSLM4MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, DenseSLM4MoeTopkRouter):
            module.weight.data.normal_(mean=0.0, std=std)
            module.e_score_correction_bias.data.zero_()
        elif isinstance(module, DenseSLM4MoeExperts):
            module.up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.RMSNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)


class DenseMLAMoeBlock(nn.Module):
    """MLA attention block followed by a shared+routed MoE FFN."""

    def __init__(self, config: DenseSLM4MoeConfig, layer_idx: int) -> None:
        super().__init__()
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
        self.moe = DenseSLM4Moe(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, _ = self.self_attn(hidden_states, attention_mask=None)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.moe(hidden_states)
        return hidden_states

    @torch.no_grad()
    def update_moe_bias(self) -> None:
        self.moe.update_expert_bias()


class DenseSLM4MoeModel(DenseSLM4MoePreTrainedModel):
    """DenseSLM4 backbone with MoE only on MLA layers; Mamba layers stay pure Mamba."""

    def __init__(self, config: DenseSLM4MoeConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            if (i + 1) % 4 == 0:
                self.layers.append(DenseMLAMoeBlock(config, layer_idx=i))
            else:
                self.layers.append(DenseMamba2Block(config, layer_idx=i))
        self.norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    @torch.no_grad()
    def update_moe_biases(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "update_moe_bias"):
                layer.update_moe_bias()


class DenseSLM4MoeForCausalLM(DenseSLM4MoePreTrainedModel, GenerationMixin):
    """DenseSLM4MoE language model with Hugging Face checkpoint compatibility."""

    def __init__(self, config: DenseSLM4MoeConfig) -> None:
        super().__init__(config)
        self.use_projected_embedding = getattr(config, "use_projected_embedding", False)
        shared_embedding_weight = None

        if self.use_projected_embedding:
            proj_path = getattr(config, "projected_embedding_path", None)
            if proj_path is not None:
                checkpoint = torch.load(proj_path, map_location="cpu")
                reduced_emb = checkpoint["reduced_embedding"]
                if reduced_emb.shape[0] != config.vocab_size:
                    print(f"Warning: projected embedding vocab_size ({reduced_emb.shape[0]}) != config vocab_size ({config.vocab_size})")
                    reduced_emb = reduced_emb[: min(reduced_emb.shape[0], config.vocab_size)]
                shared_embedding_weight = nn.Parameter(reduced_emb.float()[:, : config.hidden_size], requires_grad=False)
                print(f"Loaded projected embedding, shape: {shared_embedding_weight.shape}")

        self.model = DenseSLM4MoeModel(config)

        if self.use_projected_embedding and shared_embedding_weight is not None:
            self.model.embed_tokens = nn.Embedding.from_pretrained(shared_embedding_weight)
            for param in self.model.embed_tokens.parameters():
                param.requires_grad = False
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
            self.lm_head.weight.requires_grad = False
            print("Frozen embedding and lm_head tied together")
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def _untie_weights(self) -> None:
        if self.use_projected_embedding:
            return
        if self.lm_head.weight is self.model.embed_tokens.weight:
            self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight.data.clone())

    def _retie_weights(self) -> None:
        if self.use_projected_embedding:
            return
        if self.lm_head.weight is not self.model.embed_tokens.weight:
            self.lm_head.weight = self.model.embed_tokens.weight

    def save_pretrained(self, save_directory, **kwargs):
        self._untie_weights()
        super().save_pretrained(save_directory, **kwargs)
        self._retie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    @torch.no_grad()
    def update_moe_biases(self) -> None:
        self.model.update_moe_biases()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        **_: object,
    ) -> CausalLMOutput:
        hidden_states = self.model(input_ids=input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)


