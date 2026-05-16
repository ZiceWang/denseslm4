"""Hybrid Datasets pretraining entrypoint for DenseSLM4MoE."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from denseslm4 import DenseSLM4MoeConfig, DenseSLM4MoeForCausalLM
from denseslm4.muon import SingleDeviceMuonWithAuxAdam
from denseslm4.pretrained_dataset import load_pretrained_dataset

DEFAULT_TOKENIZER = "tokenizer_workspace"
DEFAULT_OUTPUT_DIR = Path("runs/denseslm4_moe")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def configure_torch_speed(tf32: bool) -> None:
    """Enable safe CUDA matmul speedups before model construction/training."""

    if not torch.cuda.is_available():
        return
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def prepare_causal_lm_dataset(
    raw_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    tokenize_num_proc: int | None,
    map_batch_size: int,
    text_column: str,
) -> DatasetDict:
    """Tokenize raw text and rely on datasets fingerprint caching for reuse."""

    column_name = text_column
    if column_name not in raw_dataset["train"].column_names:
        available = ", ".join(raw_dataset["train"].column_names)
        raise ValueError(f"Text column '{column_name}' not found in train split. Available columns: {available}")

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch[column_name],
            truncation=True,
            max_length=block_size,
            padding=False,
        )

    return raw_dataset.map(
        tokenize,
        batched=True,
        batch_size=map_batch_size,
        num_proc=tokenize_num_proc,
        remove_columns=raw_dataset["train"].column_names,
        desc=f"Tokenizing to {block_size} tokens",
    )


def build_model(
    tokenizer: PreTrainedTokenizerBase, 
    block_size: int,
    vocab_size: int,
    num_hidden_layers: int,
    hidden_size: int,
    num_attention_heads: int,
    moe_intermediate_size: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    moe_shared_expert_intermediate_size: int,
    n_group: int,
    topk_group: int,
    router_bias_update_rate: float,
    router_score_func: str,
    use_projected_embedding: bool = False,
    projected_embedding_path: str | None = None,
) -> DenseSLM4MoeForCausalLM:
    """Construct the default DenseSLM4MoE model for Hybrid Datasets pretraining."""

    config = DenseSLM4MoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=moe_intermediate_size,
        max_position_embeddings=block_size,
        mamba_n_groups=4,
        dropout=0.0,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        n_group=n_group,
        topk_group=topk_group,
        router_bias_update_rate=router_bias_update_rate,
        router_score_func=router_score_func,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Projected embedding options
        use_projected_embedding=use_projected_embedding,
        projected_embedding_path=projected_embedding_path,
    )
    return DenseSLM4MoeForCausalLM(config)


def ppl(loss: float) -> float:
    """Convert cross-entropy loss to perplexity."""

    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def main(
    output_dir: Annotated[Path, typer.Option(help="Directory for checkpoints and final artifacts.")] = DEFAULT_OUTPUT_DIR,
    tokenizer_name: Annotated[str, typer.Option(help="Tokenizer name or local tokenizer path.")] = DEFAULT_TOKENIZER,
    block_size: Annotated[int, typer.Option(min=8, help="Maximum sequence length before dynamic batch padding.")] = 1024,
    num_hidden_layers: Annotated[int, typer.Option(min=1, help="Number of DenseSLM4MoE backbone layers.")] = 48,
    hidden_size: Annotated[int, typer.Option(min=1, help="Model hidden size.")] = 768,
    num_attention_heads: Annotated[int, typer.Option(min=1, help="Number of MLA attention heads.")] = 16,
    moe_intermediate_size: Annotated[int, typer.Option(min=1, help="Routed expert hidden size.")] = 1024,
    n_routed_experts: Annotated[int, typer.Option(min=1, help="Number of routed experts per MoE layer.")] = 64,
    num_experts_per_tok: Annotated[int, typer.Option(min=1, help="Activated routed experts per token.")] = 2,
    moe_shared_expert_intermediate_size: Annotated[int, typer.Option(min=1, help="Shared expert hidden size.")] = 1024,
    n_group: Annotated[int, typer.Option(min=1, help="Expert groups for Nemotron/DeepSeek grouped routing.")] = 8,
    topk_group: Annotated[int, typer.Option(min=1, help="Selected expert groups per token before expert top-k.")] = 2,
    router_bias_update_rate: Annotated[float, typer.Option(min=0.0, help="Aux-free expert-bias load-balance update rate.")] = 1e-3,
    router_score_func: Annotated[str, typer.Option(help="Router score function: sigmoid or softmax.")] = "sigmoid",
    num_train_epochs: Annotated[float, typer.Option(min=0.0, help="Number of training epochs.")] = 1.0,
    batch_size: Annotated[int, typer.Option(min=1, help="Per-device train/eval batch size.")] = 32,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1, help="Gradient accumulation steps.")] = 1,
    learning_rate: Annotated[float, typer.Option(min=0.0, help="AdamW learning rate.")] = 3e-4,
    weight_decay: Annotated[float, typer.Option(min=0.0, help="AdamW weight decay.")] = 0.01,
    warmup_steps: Annotated[int, typer.Option(min=0, help="Number of scheduler warmup steps.")] = 500,
    bf16: Annotated[bool, typer.Option("--bf16/--no-bf16", help="Enable bf16 training when supported.")] = True,
    tf32: Annotated[bool, typer.Option("--tf32/--no-tf32", help="Enable TF32 matmul speedups on NVIDIA GPUs.")] = True,
    compile_model: Annotated[bool, typer.Option("--compile-model/--no-compile-model", help="Compile model with torch.compile before Trainer.")] = False,
    compile_mode: Annotated[str, typer.Option(help="torch.compile mode: default, reduce-overhead, max-autotune.")] = "reduce-overhead",
    gradient_checkpointing: Annotated[bool, typer.Option(help="Enable activation checkpointing.")] = False,
    tokenize_num_proc: Annotated[int, typer.Option(min=1, help="Tokenizer map workers.")] = os.cpu_count() or 1,
    map_batch_size: Annotated[int, typer.Option(min=1, help="Examples per datasets.map batch.")] = 1000,
    logging_steps: Annotated[int, typer.Option(min=1, help="Trainer logging interval in optimizer steps.")] = 50,
    save_total_limit: Annotated[int, typer.Option(min=1, help="Maximum number of epoch checkpoints to keep.")] = 3,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
    overwrite_output_dir: Annotated[bool, typer.Option("--overwrite-output-dir", help="Delete output_dir before training.")] = False,
    use_projected_embedding: Annotated[bool, typer.Option("--use-projected-embedding", help="Use projected frozen embedding from SVD.")] = False,
    projected_embedding_path: Annotated[str | None, typer.Option(help="Path to projected embedding checkpoint.")] = None,
) -> None:
    """Pretrain DenseSLM4MoE on Hybrid Datasets using epoch-based Trainer scheduling."""

    if router_score_func not in {"sigmoid", "softmax"}:
        raise typer.BadParameter("router_score_func must be 'sigmoid' or 'softmax'")
    if num_experts_per_tok > n_routed_experts:
        raise typer.BadParameter("num_experts_per_tok must be <= n_routed_experts")
    if n_routed_experts % n_group != 0:
        raise typer.BadParameter("n_routed_experts must be divisible by n_group")
    if topk_group > n_group:
        raise typer.BadParameter("topk_group must be <= n_group")
    if compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
        raise typer.BadParameter("compile_mode must be one of: default, reduce-overhead, max-autotune")

    set_seed(seed)
    configure_torch_speed(tf32)
    if output_dir.exists() and overwrite_output_dir:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get vocab_size from tokenizer directly
    vocab_size = len(tokenizer)
    print(f"Tokenizer: {tokenizer_name}")
    print(f"  Tokenizer vocab_size: {vocab_size}")

    # Load pretrained dataset (直接调用，无需手动指定或命令行输入)
    pretrained_dataset, text_column = load_pretrained_dataset()
    # Split into train/validation (99/1)
    split = pretrained_dataset.train_test_split(test_size=0.001, seed=seed)
    raw_dataset = DatasetDict(train=split["train"], validation=split["test"])
    train_dataset = prepare_causal_lm_dataset(
        raw_dataset,
        tokenizer,
        block_size,
        tokenize_num_proc,
        map_batch_size,
        text_column,
    )
    if len(train_dataset["train"]) == 0 or len(train_dataset["validation"]) == 0:
        raise RuntimeError("Prepared dataset is empty; decrease --block-size or increase sample count.")

    model = build_model(
        tokenizer, 
        block_size,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        moe_intermediate_size=moe_intermediate_size,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_shared_expert_intermediate_size=moe_shared_expert_intermediate_size,
        n_group=n_group,
        topk_group=topk_group,
        router_bias_update_rate=router_bias_update_rate,
        router_score_func=router_score_func,
        use_projected_embedding=use_projected_embedding,
        projected_embedding_path=projected_embedding_path,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(
        "MoE config: "
        f"shared={hidden_size}->{moe_shared_expert_intermediate_size}->{hidden_size}, "
        f"routed_experts={n_routed_experts}, top_k={num_experts_per_tok}, "
        f"groups={n_group}, topk_group={topk_group}, "
        f"expert={hidden_size}->{moe_intermediate_size}->{hidden_size}, "
        f"router={router_score_func}, bias_update={router_bias_update_rate}"
    )
    if use_projected_embedding:
        print("Using projected frozen embedding mode")
    if compile_model:
        print(f"Compiling model with torch.compile(mode='{compile_mode}')")
        model = torch.compile(model, mode=compile_mode)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        bf16=bf16 and torch.cuda.is_available(),
        tf32=tf32 and torch.cuda.is_available(),
        gradient_checkpointing=gradient_checkpointing,
        torch_empty_cache_steps=100,
        eval_strategy="steps",
        eval_steps=30000,
        save_strategy="steps",
        save_steps=10000,
        logging_steps=logging_steps,
        logging_dir=str(output_dir / "logs"),
        save_total_limit=save_total_limit,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        seed=seed,
        dataloader_drop_last=True,  # 防止最后一个batch的尺寸不对齐导致Mamba CUDA算子抛错
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    # Build Muon optimizer: hidden weight matrices -> Muon, embeddings/lm_head/gains/biases -> AdamW
    body = model.model  # DenseSLM4MoeModel (the backbone)
    
    if use_projected_embedding:
        # When using projected embedding:
        # - embed_tokens (normal embedding) is NOT used
        # - projected_embedding_layer is frozen
        # - lm_head is frozen
        # So only train body layers (Mamba/MLA/MoE blocks + norm)
        
        # Separate into hidden weights (Muon) and others (AdamW)
        hidden_weights = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
        hidden_gains_biases = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
            dict(params=hidden_gains_biases, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        print(f"Optimizer: training only body layers (embedding and lm_head frozen)")
        print(f"  Trainable params: {sum(p.numel() for p in hidden_weights + hidden_gains_biases) / 1e6:.2f}M")
    else:
        # Normal mode: train everything
        # Hidden weight matrices (2D, excluding embed_tokens which is handled separately)
        hidden_weights = [
            p
            for n, p in body.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in n and "e_score_correction_bias" not in n
        ]
        # Gains and biases (1D) + head + embeddings -> AdamW
        hidden_gains_biases = [
            p
            for n, p in body.named_parameters()
            if p.ndim < 2 and "e_score_correction_bias" not in n
        ]
        nonhidden_params = [*model.lm_head.parameters(), *body.embed_tokens.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
            dict(params=hidden_gains_biases + nonhidden_params, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
    trainer.optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    trainer.train()
    metrics = trainer.evaluate()
    metrics["perplexity"] = ppl(metrics["eval_loss"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)

    # reloaded = DenseSLM4ForCausalLM.from_pretrained(final_dir).cuda()
    # sample = torch.tensor([train_dataset["validation"][0]["input_ids"]], dtype=torch.long).cuda()
    # actual_len = sample.shape[1]
    # with torch.no_grad():
    #     logits = reloaded(sample).logits
    # expected_shape = (1, actual_len, len(tokenizer))
    # if tuple(logits.shape) != expected_shape:
    #     raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)} != {expected_shape}")

    typer.echo(f"Training complete. eval_loss={metrics['eval_loss']:.4f} ppl={metrics['perplexity']:.2f}")
    typer.echo(f"Final model: {final_dir}")


def cli() -> None:
    """Run the Hybrid Datasets pretraining CLI."""

    typer.run(main)


if __name__ == "__main__":
    cli()
