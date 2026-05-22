"""Hybrid Datasets pretraining entrypoint for DenseSLM4."""

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

from denseslm4 import DenseSLM4Config, DenseSLM4ForCausalLM
from denseslm4.muon import SingleDeviceMuonWithAuxAdam
from denseslm4.pretrained_dataset import load_pretrained_dataset

DEFAULT_TOKENIZER = "tokenizer_workspace"
DEFAULT_OUTPUT_DIR = Path("runs/my_model_new_tokenizer")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def prepare_causal_lm_dataset(
    raw_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    tokenize_num_proc: int | None,
    map_batch_size: int,
    text_column: str,
    stride: int | None = None,
    max_chunks_per_doc: int | None = None,
) -> DatasetDict:
    column_name = text_column
    if column_name not in raw_dataset["train"].column_names:
        available = ", ".join(raw_dataset["train"].column_names)
        raise ValueError(f"Text column '{column_name}' not found in train split. Available columns: {available}")

    tokenized = raw_dataset.map(
        lambda batch: tokenizer(
            batch[column_name],
            truncation=stride is None,
            max_length=block_size if stride is None else None,
            padding=False,
        ),
        batched=True,
        batch_size=map_batch_size,
        num_proc=tokenize_num_proc,
        remove_columns=raw_dataset["train"].column_names,
        desc="Tokenizing",
    )

    if stride is not None and stride < block_size:
        if "attention_mask" in tokenized["train"].column_names:
            tokenized = tokenized.remove_columns("attention_mask")

        def chunk(examples):
            chunks = []
            for ids in examples["input_ids"]:
                count = 0
                for start in range(0, len(ids) - block_size + 1, stride):
                    if max_chunks_per_doc is not None and count >= max_chunks_per_doc:
                        break
                    chunks.append(ids[start:start + block_size])
                    count += 1
            return {"input_ids": chunks}

        desc = f"Chunking stride={stride}"
        if max_chunks_per_doc is not None:
            desc += f" max={max_chunks_per_doc}"
        tokenized = tokenized.map(
            chunk,
            batched=True,
            batch_size=map_batch_size,
            num_proc=tokenize_num_proc,
            desc=desc,
        )

    return tokenized

def build_model(
    tokenizer: PreTrainedTokenizerBase, 
    block_size: int,
    vocab_size: int,
    use_projected_embedding: bool = False,
    projected_embedding_path: str | None = None,
) -> DenseSLM4ForCausalLM:
    """Construct the default DenseSLM4 model for Hybrid Datasets pretraining."""

    config = DenseSLM4Config(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=48,
        num_attention_heads=16,
        intermediate_size=768*4,
        max_position_embeddings=block_size,
        mamba_n_groups=4,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Projected embedding options
        use_projected_embedding=use_projected_embedding,
        projected_embedding_path=projected_embedding_path,
    )
    return DenseSLM4ForCausalLM(config)


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
    num_train_epochs: Annotated[float, typer.Option(min=0.0, help="Number of training epochs.")] = 1.0,
    batch_size: Annotated[int, typer.Option(min=1, help="Per-device train/eval batch size.")] = 32,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1, help="Gradient accumulation steps.")] = 1,
    learning_rate: Annotated[float, typer.Option(min=0.0, help="AdamW learning rate.")] = 3e-4,
    weight_decay: Annotated[float, typer.Option(min=0.0, help="AdamW weight decay.")] = 0.01,
    warmup_steps: Annotated[int, typer.Option(min=0, help="Number of scheduler warmup steps.")] = 0,
    bf16: Annotated[bool, typer.Option("--bf16/--no-bf16", help="Enable bf16 training when supported.")] = True,
    gradient_checkpointing: Annotated[bool, typer.Option(help="Enable activation checkpointing.")] = False,
    tokenize_num_proc: Annotated[int, typer.Option(min=1, help="Tokenizer map workers.")] = os.cpu_count() or 1,
    map_batch_size: Annotated[int, typer.Option(min=1, help="Examples per datasets.map batch.")] = 1000,
    logging_steps: Annotated[int, typer.Option(min=1, help="Trainer logging interval in optimizer steps.")] = 50,
    save_total_limit: Annotated[int, typer.Option(min=1, help="Maximum number of epoch checkpoints to keep.")] = 3,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
    overwrite_output_dir: Annotated[bool, typer.Option("--overwrite-output-dir", help="Delete output_dir before training.")] = False,
    use_projected_embedding: Annotated[bool, typer.Option("--use-projected-embedding", help="Use projected frozen embedding from SVD.")] = False,
    projected_embedding_path: Annotated[str | None, typer.Option(help="Path to projected embedding checkpoint.")] = None,
    stride: Annotated[int | None, typer.Option("--stride", help="Sliding window stride for chunking (None = truncate to block_size).")] = None,
    muon_scale: Annotated[str, typer.Option("--muon-scale", help="Muon update scaling: kellerjordan or moonlight.")] = "moonlight",
    max_chunks_per_doc: Annotated[int | None, typer.Option("--max-chunks-per-doc", help="Max chunks per document with stride (None = unlimited).")] = None,
) -> None:
    """Pretrain DenseSLM4 on Hybrid Datasets using epoch-based Trainer scheduling."""

    set_seed(seed)
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
        stride=stride,
        max_chunks_per_doc=max_chunks_per_doc,
    )
    if len(train_dataset["train"]) == 0 or len(train_dataset["validation"]) == 0:
        raise RuntimeError("Prepared dataset is empty; decrease --block-size or increase sample count.")

    overlap = block_size - stride if (stride is not None and stride < block_size) else 0
    print(f"Data: {len(train_dataset['train'])} train / {len(train_dataset['validation'])} val samples")
    cap = f", max_chunks={max_chunks_per_doc}" if max_chunks_per_doc else ""
    print(f"Chunk: block_size={block_size}, stride={stride or block_size}, overlap={overlap}{cap}")

    model = build_model(
        tokenizer, 
        block_size,
        vocab_size=vocab_size,
        use_projected_embedding=use_projected_embedding,
        projected_embedding_path=projected_embedding_path,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    if use_projected_embedding:
        print("Using projected frozen embedding mode")
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
        gradient_checkpointing=gradient_checkpointing,
        torch_empty_cache_steps=100,
        eval_strategy="steps",
        eval_steps=1000,
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
    body = model.model  # DenseSLM4Model (the backbone)
    
    if use_projected_embedding:
        # When using projected embedding:
        # - embed_tokens (normal embedding) is NOT used
        # - projected_embedding_layer is frozen
        # - lm_head is frozen
        # So only train body layers (MLM/Mamba blocks + norm)
        trainable_params = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                trainable_params.append(p)
        
        # Separate into hidden weights (Muon) and others (AdamW)
        hidden_weights = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
        hidden_gains_biases = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01, scale_mode=muon_scale),
            dict(params=hidden_gains_biases, use_muon=False, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
        ]
        print(f"Optimizer: training only body layers (embedding and lm_head frozen)")
        print(f"  Trainable params: {sum(p.numel() for p in hidden_weights + hidden_gains_biases) / 1e6:.2f}M")
    else:
        # Normal mode: train everything
        # Hidden weight matrices (2D, excluding embed_tokens which is handled separately)
        hidden_weights = [p for n, p in body.named_parameters() if p.ndim >= 2 and "embed_tokens" not in n]
        # Gains and biases (1D) + head + embeddings -> AdamW
        hidden_gains_biases = [p for n, p in body.named_parameters() if p.ndim < 2]
        nonhidden_params = [*model.lm_head.parameters(), *body.embed_tokens.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01, scale_mode=muon_scale),
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
