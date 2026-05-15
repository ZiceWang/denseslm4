"""Minimal DenseSLM4 modeling code built on PyTorch/Transformers primitives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput

from .configuration_denseslm4 import DenseSLM4Config
from .modules import DenseMLABlock, DenseDeltaFormerBlock, DenseLatentDeltaFormerBlock, DenseMamba2Block,DenseMLAIHCBlock

class DenseSLM4PreTrainedModel(PreTrainedModel):
    """Base class with Transformers-compatible initialization."""

    config_class = DenseSLM4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True

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
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.RMSNorm):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)


class DenseSLM4Model(DenseSLM4PreTrainedModel):
    """A compact decoder-only Transformer backbone with rotary position embeddings."""

    def __init__(self, config: DenseSLM4Config) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList()
        # 3 layers Mamba3 + 1 layer DeepSeekV3 MLA (with RoPE)
        for i in range(config.num_hidden_layers):
            if (i + 1) % 4 == 0:
                self.layers.append(DenseMLABlock(config, layer_idx=i))
            else:
                self.layers.append(DenseMamba2Block(config, layer_idx=i))
                
        self.norm = nn.LayerNorm(config.hidden_size)
        self.post_init()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if getattr(self.config, "max_position_embeddings", 0) > 0 and seq_len > self.config.max_position_embeddings:
            pass # No strict cutoff needed for NOPE / Mamba, they extrapolate gracefully

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return self.norm(hidden_states)


class DenseSLM4ForCausalLM(DenseSLM4PreTrainedModel, GenerationMixin):
    """DenseSLM4 language model with Hugging Face checkpoint compatibility."""

    def __init__(self, config: DenseSLM4Config) -> None:
        super().__init__(config)
        
        # Handle projected embedding mode: load shared embedding weight first
        self.use_projected_embedding = getattr(config, "use_projected_embedding", False)
        shared_embedding_weight = None
        
        if self.use_projected_embedding:
            proj_path = getattr(config, "projected_embedding_path", None)
            if proj_path is not None:
                checkpoint = torch.load(proj_path, map_location="cpu")
                proj_vocab_size = checkpoint["reduced_embedding"].shape[0]
                config_vocab_size = config.vocab_size
                
                if proj_vocab_size != config_vocab_size:
                    print(f"Warning: projected embedding vocab_size ({proj_vocab_size}) != config vocab_size ({config_vocab_size})")
                    min_vocab = min(proj_vocab_size, config_vocab_size)
                    reduced_emb = checkpoint["reduced_embedding"][:min_vocab]
                else:
                    reduced_emb = checkpoint["reduced_embedding"]
                
                # Create shared embedding weight (frozen)
                shared_embedding_weight = nn.Parameter(reduced_emb.float()[:, :config.hidden_size], requires_grad=False)
                print(f"Loaded projected embedding, shape: {shared_embedding_weight.shape}")
        
        # Initialize model (DenseSLM4Model)
        self.model = DenseSLM4Model(config)
        
        if self.use_projected_embedding and shared_embedding_weight is not None:
            # Use projected embedding for both embed_tokens and lm_head (tied)
            self.model.embed_tokens = nn.Embedding.from_pretrained(shared_embedding_weight)
            # Freeze it
            for param in self.model.embed_tokens.parameters():
                param.requires_grad = False
            
            # lm_head shares the SAME weight as embed_tokens (tied)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # TIE!
            self.lm_head.weight.requires_grad = False
            print(f"Frozen embedding and lm_head tied together")
        else:
            # Normal mode: tie embedding and lm_head
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # tie weights
        self.post_init()

    def _untie_weights(self):
        """Untie lm_head from embed_tokens before saving."""
        if self.use_projected_embedding:
            # In projected mode, embedding is frozen and tied - no need to untie for saving
            return
        if self.lm_head.weight is self.model.embed_tokens.weight:
            self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight.data.clone())

    def _retie_weights(self):
        """Retie lm_head to embed_tokens after loading."""
        if self.use_projected_embedding:
            # In projected mode, embedding is frozen and tied - no need to retie
            return
        if self.lm_head.weight is not self.model.embed_tokens.weight:
            self.lm_head.weight = self.model.embed_tokens.weight

    def save_pretrained(self, save_directory, **kwargs):
        """Save pretrained model, handling tied weights."""
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
