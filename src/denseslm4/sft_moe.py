"""Supervised fine-tuning entrypoint for DenseSLM4MoE chat datasets."""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase, set_seed
from trl import SFTConfig, SFTTrainer

from denseslm4 import DenseSLM4MoeForCausalLM

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_MODEL = Path("runs/denseslm4_moe_v2/final_model")
DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_OUTPUT_DIR = Path("runs/denseslm4_moe_v2_sft_ultrachat_200k")
DEFAULT_CHAT_TEMPLATE = """{%- for message in messages %}
{{- bos_token + message['role'] + '\n' }}
{{- message['content'] | trim + eos_token + '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- bos_token + 'assistant\n' }}
{%- endif %}
"""


def configure_torch_speed(tf32: bool) -> None:
	if not torch.cuda.is_available():
		return
	if tf32:
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True
		torch.set_float32_matmul_precision("high")


def _message_text(message: dict[str, Any]) -> str:
	content = message.get("content", "")
	if content is None:
		content = ""
	return str(content).strip()


def normalize_messages(example: dict[str, Any]) -> list[dict[str, str]]:
	"""Convert common SFT schemas into OpenAI-style chat messages."""

	if example.get("messages"):
		return [
			{"role": str(message.get("role", "user")).strip().lower(), "content": _message_text(message)}
			for message in example["messages"]
			if _message_text(message)
		]
	if example.get("conversations"):
		messages: list[dict[str, str]] = []
		for turn in example["conversations"]:
			role = str(turn.get("role") or turn.get("from") or "user").strip().lower()
			if role in {"human", "user"}:
				role = "user"
			elif role in {"gpt", "assistant", "model"}:
				role = "assistant"
			elif role != "system":
				role = "user"
			content = str(turn.get("content") or turn.get("value") or "").strip()
			if content:
				messages.append({"role": role, "content": content})
		return messages

	instruction = str(example.get("instruction") or example.get("prompt") or example.get("question") or "").strip()
	input_text = str(example.get("input") or "").strip()
	output = str(example.get("output") or example.get("response") or example.get("answer") or "").strip()
	if instruction and input_text:
		instruction = f"{instruction}\n\n{input_text}"
	if instruction and output:
		return [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
	return []


def has_assistant_message(example: dict[str, Any]) -> bool:
	return any(message["role"] == "assistant" and message["content"] for message in normalize_messages(example))


def _format_message(role: str, content: str, bos_token: str, eos_token: str) -> str:
	return f"{bos_token}{role}\n{content}{eos_token}\n"


def ensure_chat_template(tokenizer: PreTrainedTokenizerBase) -> None:
	"""Install the DenseSLM4 chat template when the tokenizer does not ship one."""

	if tokenizer.chat_template:
		return
	tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE


def render_messages(messages: list[dict[str, Any]], bos_token: str = "<|im_start|>", eos_token: str = "<|im_end|>") -> tuple[str, str]:
	"""Render a chat sample as repeated BOS/EOS role spans.

	Returns `(prompt_text, answer_text)`, so label masking is token-count based instead of char-slice based.
	"""

	prompt_chunks: list[str] = []
	answer_chunks: list[str] = []
	assistant_seen = False
	for message in messages:
		role = str(message.get("role", "user")).strip().lower()
		content = _message_text(message)
		if role == "assistant":
			assistant_seen = True
			answer_chunks.append(_format_message("assistant", content, bos_token, eos_token))
		elif role == "system":
			prompt_chunks.append(_format_message("system", content, bos_token, eos_token))
		else:
			prompt_chunks.append(_format_message("user", content, bos_token, eos_token))

	if not assistant_seen:
		raise ValueError("SFT example has no assistant message")
	return "".join(prompt_chunks), "".join(answer_chunks)


def render_input_and_labels(
	messages: list[dict[str, Any]],
	tokenizer: PreTrainedTokenizerBase,
	max_length: int,
	bos_token: str,
	eos_token: str,
) -> tuple[list[int], list[int]]:
	"""Render all turns while masking non-assistant tokens."""

	input_ids: list[int] = []
	labels: list[int] = []
	assistant_seen = False
	for message in messages:
		role = str(message.get("role", "user")).strip().lower()
		content = _message_text(message)
		if not content:
			continue
		if role not in {"system", "user", "assistant"}:
			role = "user"
		turn_ids = tokenizer(_format_message(role, content, bos_token, eos_token), add_special_tokens=False)["input_ids"]
		input_ids.extend(turn_ids)
		if role == "assistant":
			assistant_seen = True
			labels.extend(turn_ids)
		else:
			labels.extend([-100] * len(turn_ids))

	if not assistant_seen:
		raise ValueError("SFT example has no assistant message")
	input_ids = input_ids[-max_length:]
	labels = labels[-max_length:]
	if all(label == -100 for label in labels):
		labels[-1] = input_ids[-1]
	return input_ids, labels


def tokenize_sft_dataset(
	dataset: Dataset,
	tokenizer: PreTrainedTokenizerBase,
	max_length: int,
	num_proc: int | None,
	map_batch_size: int,
) -> Dataset:
	ensure_chat_template(tokenizer)
	bos_token = tokenizer.bos_token or "<|im_start|>"
	eos_token = tokenizer.eos_token or "<|im_end|>"

	def convert(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
		input_ids_batch: list[list[int]] = []
		attention_mask_batch: list[list[int]] = []
		labels_batch: list[list[int]] = []

		batch_size = len(next(iter(batch.values())))
		for index in range(batch_size):
			example = {key: values[index] for key, values in batch.items()}
			input_ids, labels = render_input_and_labels(normalize_messages(example), tokenizer, max_length, bos_token, eos_token)
			input_ids_batch.append(input_ids)
			attention_mask_batch.append([1] * len(input_ids))
			labels_batch.append(labels)

		return {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels_batch}

	return dataset.map(
		convert,
		batched=True,
		batch_size=map_batch_size,
		num_proc=num_proc,
		remove_columns=dataset.column_names,
		desc="Tokenizing SFT dataset",
	)


@dataclass
class SFTDataCollator:
	tokenizer: PreTrainedTokenizerBase
	pad_to_multiple_of: int = 8

	def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
		labels = [feature.pop("labels") for feature in features]
		batch = self.tokenizer.pad(
			features,
			padding=True,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)
		max_len = batch["input_ids"].shape[1]
		padded_labels = []
		for label in labels:
			pad_len = max_len - len(label)
			padded_labels.append(label + [-100] * pad_len)
		batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
		return batch


def perplexity(loss: float) -> float:
	try:
		return math.exp(loss)
	except OverflowError:
		return float("inf")


def main(
	model_name_or_path: Annotated[Path, typer.Option(help="DenseSLM4MoE base/final model path.")] = DEFAULT_MODEL,
	dataset_name: Annotated[str, typer.Option(help="Hugging Face dataset name.")] = DEFAULT_DATASET,
	output_dir: Annotated[Path, typer.Option(help="Directory for SFT checkpoints and final model.")] = DEFAULT_OUTPUT_DIR,
	split: Annotated[str, typer.Option(help="Dataset split to train on.")] = "train_sft",
	max_length: Annotated[int, typer.Option(min=64, help="Maximum sequence length.")] = 1024,
	num_train_epochs: Annotated[float, typer.Option(min=0.0, help="Number of SFT epochs.")] = 1.0,
	batch_size: Annotated[int, typer.Option(min=1, help="Per-device train/eval batch size.")] = 2,
	gradient_accumulation_steps: Annotated[int, typer.Option(min=1, help="Gradient accumulation steps.")] = 16,
	learning_rate: Annotated[float, typer.Option(min=0.0, help="Fine-tuning learning rate.")] = 2e-5,
	weight_decay: Annotated[float, typer.Option(min=0.0, help="Weight decay.")] = 0.0,
	warmup_ratio: Annotated[float, typer.Option(min=0.0, max=1.0, help="Warmup ratio.")] = 0.03,
	validation_ratio: Annotated[float, typer.Option(min=0.0, max=0.5, help="Validation split ratio.")] = 0.02,
	bf16: Annotated[bool, typer.Option("--bf16/--no-bf16", help="Enable bf16 on CUDA.")] = True,
	tf32: Annotated[bool, typer.Option("--tf32/--no-tf32", help="Enable TF32 matmul speedups.")] = True,
	gradient_checkpointing: Annotated[bool, typer.Option("--gradient-checkpointing/--no-gradient-checkpointing")] = True,
	tokenize_num_proc: Annotated[int, typer.Option(min=1, help="Tokenizer map workers.")] = min(os.cpu_count() or 1, 8),
	map_batch_size: Annotated[int, typer.Option(min=1, help="Examples per datasets.map batch.")] = 512,
	logging_steps: Annotated[int, typer.Option(min=1, help="Logging interval.")] = 10,
	save_steps: Annotated[int, typer.Option(min=1, help="Checkpoint interval.")] = 200,
	eval_steps: Annotated[int, typer.Option(min=1, help="Eval interval.")] = 200,
	save_total_limit: Annotated[int, typer.Option(min=1, help="Maximum checkpoints to keep.")] = 2,
	seed: Annotated[int, typer.Option(help="Random seed.")] = 7,
	overwrite_output_dir: Annotated[bool, typer.Option("--overwrite-output-dir", help="Delete output dir first.")] = False,
) -> None:
	set_seed(seed)
	configure_torch_speed(tf32)
	if output_dir.exists() and overwrite_output_dir:
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	ensure_chat_template(tokenizer)

	raw = load_dataset(dataset_name, split=split, verification_mode="no_checks")
	raw = raw.filter(has_assistant_message, desc="Filtering SFT conversations")
	split_dataset = raw.train_test_split(test_size=validation_ratio, seed=seed) if validation_ratio > 0 else {"train": raw, "test": raw.select(range(min(128, len(raw))))}
	train_dataset = tokenize_sft_dataset(split_dataset["train"], tokenizer, max_length, tokenize_num_proc, map_batch_size)
	eval_dataset = tokenize_sft_dataset(split_dataset["test"], tokenizer, max_length, tokenize_num_proc, map_batch_size)

	model = DenseSLM4MoeForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16 if bf16 and torch.cuda.is_available() else None)
	model.config.use_cache = False
	if gradient_checkpointing:
		model.gradient_checkpointing_enable()

	training_args = SFTConfig(
		output_dir=str(output_dir),
		num_train_epochs=num_train_epochs,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		gradient_accumulation_steps=gradient_accumulation_steps,
		learning_rate=learning_rate,
		weight_decay=weight_decay,
		warmup_steps=max(1, int((len(train_dataset) // max(1, batch_size * gradient_accumulation_steps)) * num_train_epochs * warmup_ratio)),
		bf16=bf16 and torch.cuda.is_available(),
		tf32=tf32 and torch.cuda.is_available(),
		gradient_checkpointing=gradient_checkpointing,
		gradient_checkpointing_kwargs={"use_reentrant": False},
		lr_scheduler_type="cosine",
		eval_strategy="steps",
		eval_steps=eval_steps,
		save_strategy="steps",
		save_steps=save_steps,
		logging_steps=logging_steps,
		save_total_limit=save_total_limit,
		report_to=["tensorboard"],
		remove_unused_columns=False,
		seed=seed,
		dataloader_drop_last=True,
		max_length=max_length,
		pad_to_multiple_of=8,
		packing=False,
		completion_only_loss=False,
		assistant_only_loss=False,
		loss_type="nll",
		dataset_kwargs={"skip_prepare_dataset": True},
	)
	trainer = SFTTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=SFTDataCollator(tokenizer),
		processing_class=tokenizer,
	)

	trainer.train()
	metrics = trainer.evaluate()
	metrics["perplexity"] = perplexity(metrics["eval_loss"])
	trainer.log_metrics("eval", metrics)
	trainer.save_metrics("eval", metrics)

	final_dir = output_dir / "final_model"
	trainer.save_model(str(final_dir))
	tokenizer.save_pretrained(final_dir)
	typer.echo(f"SFT complete. eval_loss={metrics['eval_loss']:.4f} ppl={metrics['perplexity']:.2f}")
	typer.echo(f"Final model: {final_dir}")


if __name__ == "__main__":
	typer.run(main)
