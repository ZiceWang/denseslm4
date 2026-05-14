"""TinyStories pretraining entrypoint for DenseSLM4."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from denseslm4 import DenseSLM4Config, DenseSLM4ForCausalLM
from denseslm4.muon import SingleDeviceMuonWithAuxAdam


DEFAULT_DATASET = "nampdn-ai/tiny-textbooks"
# use tiny stories
# DEFAULT_DATASET = "nampdn-ai/tiny-stories"
DEFAULT_TOKENIZER = "Qwen/Qwen3-8B"
# DEFAULT_COLUMN = "text"
DEFAULT_COLUMN = "textbook"
DEFAULT_OUTPUT_DIR = Path("runs/tinystories")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def split_name(name: str, limit: int | None) -> str:
    """Build a Hugging Face split expression."""

    return name if limit is None else f"{name}[:{limit}]"


def load_splits(
    dataset_name: str,
    train_samples: int | None,
    eval_samples: int | None,
    data_files: list[str] | None = None,
) -> DatasetDict:
    """Load train/validation splits from HuggingFace Hub, local parquet, or cache."""

    load_kwargs: dict[str, Any] = {}
    if data_files is not None:
        load_kwargs["data_files"] = data_files
        load_kwargs["verification_mode"] = "no_checks"
        load_kwargs["split"] = None  # data_files mode doesn't support split param
        full_dataset = load_dataset(dataset_name, **load_kwargs)
        # If it returns a DatasetDict, use "train" or first dataset
        if isinstance(full_dataset, DatasetDict):
            if "train" in full_dataset:
                full_dataset = full_dataset["train"]
            else:
                full_dataset = next(iter(full_dataset.values()))
        # Split into train/validation (90/10)
        split = full_dataset.train_test_split(test_size=0.01, seed=42)
        train = split["train"]
        validation = split["test"]
    else:
        train = load_dataset(dataset_name, split=split_name("train", train_samples))
        validation = load_dataset(dataset_name, split=split_name("test", eval_samples))
    return DatasetDict(train=train, validation=validation)


def prepare_causal_lm_dataset(
    raw_dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
    tokenize_num_proc: int | None,
    map_batch_size: int,
    text_column: str | None,
) -> DatasetDict:
    """Tokenize raw text and rely on datasets fingerprint caching for reuse."""

    column_name = text_column or DEFAULT_COLUMN
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
    use_projected_embedding: bool = False,
    projected_embedding_path: str | None = None,
) -> DenseSLM4ForCausalLM:
    """Construct the default DenseSLM4 model for TinyStories pretraining."""

    config = DenseSLM4Config(
        vocab_size=vocab_size,
        hidden_size=512,
        num_hidden_layers=32,
        num_attention_heads=16,
        intermediate_size=1024,
        max_position_embeddings=block_size,
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
    dataset_name: Annotated[str, typer.Option(help="Dataset name or local dataset path.")] = DEFAULT_DATASET,
    tokenizer_name: Annotated[str, typer.Option(help="Tokenizer name or local tokenizer path.")] = DEFAULT_TOKENIZER,
    text_column: Annotated[str, typer.Option(help="Dataset column containing text.")] = DEFAULT_COLUMN,
    train_samples: Annotated[int | None, typer.Option(min=1, help="Optional train split prefix for debugging.")] = None,
    eval_samples: Annotated[int | None, typer.Option(min=1, help="Optional validation split prefix for debugging.")] = None,
    data_files: Annotated[str | None, typer.Option(help="Comma-separated list of parquet files for local datasets.")] = None,
    block_size: Annotated[int, typer.Option(min=8, help="Maximum sequence length before dynamic batch padding.")] = 128,
    num_train_epochs: Annotated[float, typer.Option(min=0.0, help="Number of training epochs.")] = 1.0,
    batch_size: Annotated[int, typer.Option(min=1, help="Per-device train/eval batch size.")] = 256,
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
) -> None:
    """Pretrain DenseSLM4 on TinyStories using epoch-based Trainer scheduling."""

    set_seed(seed)
    if output_dir.exists() and overwrite_output_dir:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get vocab_size from model config, not tokenizer (Qwen3: model=151936, tokenizer=151669)
    model_config = AutoConfig.from_pretrained(tokenizer_name)
    vocab_size = model_config.vocab_size
    print(f"Tokenizer: {tokenizer_name}")
    print(f"  Tokenizer vocab_size: {len(tokenizer)}")
    print(f"  Model config vocab_size: {vocab_size}")

    data_files_list: list[str] | None = None
    if data_files:
        data_files_list = [f.strip() for f in data_files.split(",")]
    raw_dataset = load_splits(dataset_name, train_samples, eval_samples, data_files=data_files_list)
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
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=5000,
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
            dict(params=hidden_weights, use_muon=True, lr=0.02, weight_decay=0.01),
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
    """Run the TinyStories pretraining CLI."""

    typer.run(main)


if __name__ == "__main__":
    cli()
