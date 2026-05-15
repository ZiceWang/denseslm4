# This script tests the training of low-rank MoE models with different configurations.
# All moe is aux free moe version like DeepseekV3.
# Version A: low rank expert mlp: 384->192->384, 32 expert activate 8 experts per token
# Version B: high rank expert mlp: 384->384*2->384, 8 expert activate 2 experts per token
# Version C: mid rank expert mlp: 384->384->384, 16 expert activate 4 experts per token

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import DatasetDict, load_dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_cosine_schedule_with_warmup


@dataclass(frozen=True)
class MoeSpec:
	name: str
	expert_hidden_size: int
	num_experts: int
	top_k: int


MOE_SPECS = {
	"A": MoeSpec("A_low_rank_32e_top8_h192", expert_hidden_size=192, num_experts=32, top_k=8),
	"B": MoeSpec("B_high_rank_8e_top2_h768", expert_hidden_size=384 * 2, num_experts=8, top_k=2),
	"C": MoeSpec("C_mid_rank_16e_top4_h384", expert_hidden_size=384, num_experts=16, top_k=4),
}


@dataclass
class AblationConfig:
	dataset_path: str = "./dataset/orca_math_qa.parquet"
	text_column: str = "text"
	tokenizer_name: str = "tokenizer_workspace"
	output_dir: str = "runs/moe_low_rank_ablation"
	seq_len: int = 512
	batch_size: int = 64
	num_layers: int = 4
	hidden_size: int = 384
	num_heads: int = 6
	dropout: float = 0.0
	max_steps: int = 1000
	eval_steps: int = 100
	save_steps: int = 0
	learning_rate: float = 3e-4
	weight_decay: float = 0.1
	warmup_steps: int = 100
	grad_clip: float = 1.0
	router_bias_update_rate: float = 1e-3
	router_score_func: str = "sigmoid"
	seed: int = 42
	num_workers: int = 4
	tokenize_num_proc: int = max(1, min(os.cpu_count() or 1, 16))
	map_batch_size: int = 1000
	val_fraction: float = 0.02
	max_train_samples: int | None = None
	max_eval_batches: int = 50
	bf16: bool = True
	overwrite_output_dir: bool = False


class RMSNorm(nn.Module):
	def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
	x1 = x[..., : x.shape[-1] // 2]
	x2 = x[..., x.shape[-1] // 2 :]
	return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
	def __init__(self, head_dim: int, max_position_embeddings: int, theta: float = 10000.0) -> None:
		super().__init__()
		inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
		position_ids = torch.arange(max_position_embeddings, dtype=torch.float)
		freqs = torch.outer(position_ids, inv_freq)
		emb = torch.cat((freqs, freqs), dim=-1)
		self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
		self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

	def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		seq_len = q.shape[-2]
		cos = self.cos_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
		sin = self.sin_cached[:, :, :seq_len, :].to(dtype=q.dtype, device=q.device)
		return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
	def __init__(self, hidden_size: int, num_heads: int, seq_len: int, dropout: float) -> None:
		super().__init__()
		if hidden_size % num_heads != 0:
			raise ValueError("hidden_size must be divisible by num_heads")
		self.hidden_size = hidden_size
		self.num_heads = num_heads
		self.head_dim = hidden_size // num_heads
		self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
		self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
		self.rope = RotaryEmbedding(self.head_dim, seq_len)
		self.dropout = dropout

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		bsz, seq_len, _ = x.shape
		qkv = self.qkv_proj(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
		q, k, v = qkv.unbind(dim=2)
		q = q.transpose(1, 2)
		k = k.transpose(1, 2)
		v = v.transpose(1, 2)
		q, k = self.rope(q, k)
		y = F.scaled_dot_product_attention(
			q,
			k,
			v,
			attn_mask=None,
			dropout_p=self.dropout if self.training else 0.0,
			is_causal=True,
		)
		y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
		return self.o_proj(y)


class AuxFreeTopKMoe(nn.Module):
	"""DeepSeek-V3-style aux-loss-free top-k MoE.

	The router keeps a non-trainable per-expert correction bias. Top-k expert selection
	uses ``score + bias`` while dispatch weights use the original affinity scores. The
	bias is updated after each optimizer step from observed expert load, avoiding any
	auxiliary load-balancing loss term in the LM objective.
	"""

	def __init__(
		self,
		hidden_size: int,
		expert_hidden_size: int,
		num_experts: int,
		top_k: int,
		router_bias_update_rate: float,
		router_score_func: str,
	) -> None:
		super().__init__()
		self.hidden_size = hidden_size
		self.expert_hidden_size = expert_hidden_size
		self.num_experts = num_experts
		self.top_k = top_k
		self.router_bias_update_rate = router_bias_update_rate
		if router_score_func not in {"sigmoid", "softmax"}:
			raise ValueError("router_score_func must be 'sigmoid' or 'softmax'")
		self.router_score_func = router_score_func
		self.gate = nn.Linear(hidden_size, num_experts, bias=False)
		self.register_buffer("expert_bias", torch.zeros(num_experts), persistent=True)
		self.register_buffer("last_expert_load", torch.zeros(num_experts), persistent=False)
		self.register_buffer("last_target_load", torch.tensor(0.0), persistent=False)
		self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, expert_hidden_size))
		self.w2 = nn.Parameter(torch.empty(num_experts, expert_hidden_size, hidden_size))
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.normal_(self.w1, mean=0.0, std=0.02)
		nn.init.normal_(self.w2, mean=0.0, std=0.02)
		nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		original_shape = x.shape
		flat_x = x.reshape(-1, self.hidden_size)
		router_logits = self.gate(flat_x).float()
		if self.router_score_func == "sigmoid":
			router_scores = router_logits.sigmoid()
		else:
			router_scores = router_logits.softmax(dim=-1)
		_, topk_idx = torch.topk(router_scores + self.expert_bias.view(1, -1), self.top_k, dim=-1)
		topk_scores = router_scores.gather(1, topk_idx)
		topk_weights = (topk_scores / topk_scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)).to(dtype=flat_x.dtype)

		if self.training:
			with torch.no_grad():
				load = torch.bincount(topk_idx.reshape(-1), minlength=self.num_experts).to(dtype=torch.float32, device=flat_x.device)
				self.last_expert_load.copy_(load)
				self.last_target_load.copy_(load.mean())
		out = torch.zeros_like(flat_x)

		for expert_idx in range(self.num_experts):
			token_idx, rank_idx = torch.where(topk_idx == expert_idx)
			if token_idx.numel() == 0:
				continue
			expert_in = flat_x.index_select(0, token_idx)
			hidden = F.silu(expert_in @ self.w1[expert_idx])
			expert_out = hidden @ self.w2[expert_idx]
			expert_out = expert_out * topk_weights[token_idx, rank_idx].unsqueeze(-1)
			out.index_add_(0, token_idx, expert_out)

		return out.view(original_shape)

	@torch.no_grad()
	def update_expert_bias(self) -> None:
		if self.router_bias_update_rate <= 0:
			return
		target = self.last_target_load
		if target <= 0:
			return
		load_error = torch.sign(target - self.last_expert_load)
		self.expert_bias.add_(self.router_bias_update_rate * load_error)
		self.expert_bias.sub_(self.expert_bias.mean())


class TransformerMoeBlock(nn.Module):
	def __init__(self, cfg: AblationConfig, moe_spec: MoeSpec) -> None:
		super().__init__()
		self.attn_norm = RMSNorm(cfg.hidden_size)
		self.attn = CausalSelfAttention(cfg.hidden_size, cfg.num_heads, cfg.seq_len, cfg.dropout)
		self.moe_norm = RMSNorm(cfg.hidden_size)
		self.moe = AuxFreeTopKMoe(
			hidden_size=cfg.hidden_size,
			expert_hidden_size=moe_spec.expert_hidden_size,
			num_experts=moe_spec.num_experts,
			top_k=moe_spec.top_k,
			router_bias_update_rate=cfg.router_bias_update_rate,
			router_score_func=cfg.router_score_func,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.attn_norm(x))
		x = x + self.moe(self.moe_norm(x))
		return x

	def update_moe_bias(self) -> None:
		self.moe.update_expert_bias()


class StandardTransformerMoeLM(nn.Module):
	def __init__(self, cfg: AblationConfig, vocab_size: int, pad_token_id: int, moe_spec: MoeSpec) -> None:
		super().__init__()
		self.vocab_size = vocab_size
		self.embed_tokens = nn.Embedding(vocab_size, cfg.hidden_size, padding_idx=pad_token_id)
		self.dropout = nn.Dropout(cfg.dropout)
		self.layers = nn.ModuleList([TransformerMoeBlock(cfg, moe_spec) for _ in range(cfg.num_layers)])
		self.norm = RMSNorm(cfg.hidden_size)
		self.lm_head = nn.Linear(cfg.hidden_size, vocab_size, bias=False)
		self.lm_head.weight = self.embed_tokens.weight
		self.apply(self._init_weights)

	@staticmethod
	def _init_weights(module: nn.Module) -> None:
		if isinstance(module, nn.Linear):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.padding_idx is not None:
				with torch.no_grad():
					module.weight[module.padding_idx].zero_()

	def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor | None = None) -> dict[str, torch.Tensor]:
		hidden_states = self.dropout(self.embed_tokens(input_ids))
		for layer in self.layers:
			hidden_states = layer(hidden_states)
		logits = self.lm_head(self.norm(hidden_states))
		loss = None
		if labels is not None:
			loss = F.cross_entropy(
				logits[:, :-1, :].contiguous().view(-1, self.vocab_size),
				labels[:, 1:].contiguous().view(-1),
				ignore_index=-100,
			)
		return {"loss": loss, "logits": logits}

	@torch.no_grad()
	def update_moe_biases(self) -> None:
		for layer in self.layers:
			layer.update_moe_bias()

	@torch.no_grad()
	def moe_load_stats(self) -> dict[str, float]:
		loads = torch.stack([layer.moe.last_expert_load.float() for layer in self.layers])
		if loads.numel() == 0 or float(loads.sum()) == 0.0:
			return {"moe_load_cv": 0.0, "moe_load_max_over_mean": 0.0, "moe_bias_abs_mean": 0.0}
		means = loads.mean(dim=-1).clamp_min(1e-12)
		cv = (loads.std(dim=-1, unbiased=False) / means).mean().item()
		max_over_mean = (loads.max(dim=-1).values / means).mean().item()
		bias_abs = torch.stack([layer.moe.expert_bias.abs().mean() for layer in self.layers]).mean().item()
		return {"moe_load_cv": cv, "moe_load_max_over_mean": max_over_mean, "moe_bias_abs_mean": bias_abs}


def set_reproducible_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parameter_count(model: nn.Module) -> int:
	return sum(p.numel() for p in model.parameters())


def prepare_datasets(cfg: AblationConfig):
	dataset_5 = load_dataset("parquet", split="train", data_files=cfg.dataset_path, verification_mode="no_checks")
	if cfg.max_train_samples is not None:
		dataset_5 = dataset_5.shuffle(seed=cfg.seed).select(range(min(cfg.max_train_samples, len(dataset_5))))
	if cfg.text_column not in dataset_5.column_names:
		raise ValueError(f"Text column '{cfg.text_column}' not found. Available columns: {dataset_5.column_names}")

	split = dataset_5.train_test_split(test_size=cfg.val_fraction, seed=cfg.seed)
	raw = DatasetDict(train=split["train"], validation=split["test"])
	tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	def tokenize(batch):
		return tokenizer(
			batch[cfg.text_column],
			truncation=True,
			max_length=cfg.seq_len,
			padding=False,
		)

	tokenized = raw.map(
		tokenize,
		batched=True,
		batch_size=cfg.map_batch_size,
		num_proc=cfg.tokenize_num_proc,
		remove_columns=raw["train"].column_names,
		desc=f"Tokenizing orca_math_qa to {cfg.seq_len}",
	)
	collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
	return tokenizer, tokenized, collator


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, autocast_dtype: torch.dtype | None, max_batches: int) -> float:
	model.eval()
	losses: list[float] = []
	for step, batch in enumerate(dataloader):
		if step >= max_batches:
			break
		batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "labels"}}
		with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
			loss = model(**batch)["loss"]
		losses.append(float(loss.detach().cpu()))
	model.train()
	return sum(losses) / max(1, len(losses))


def train_one_spec(cfg: AblationConfig, moe_spec: MoeSpec, tokenizer, tokenized, collator, run_dir: Path) -> list[dict[str, float | int | str]]:
	set_reproducible_seed(cfg.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	autocast_dtype = torch.bfloat16 if cfg.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported() else None

	train_loader = DataLoader(
		tokenized["train"],
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		pin_memory=device.type == "cuda",
		drop_last=True,
		collate_fn=collator,
	)
	eval_loader = DataLoader(
		tokenized["validation"],
		batch_size=cfg.batch_size,
		shuffle=False,
		num_workers=cfg.num_workers,
		pin_memory=device.type == "cuda",
		drop_last=False,
		collate_fn=collator,
	)

	model = StandardTransformerMoeLM(cfg, vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id, moe_spec=moe_spec).to(device)
	params_m = parameter_count(model) / 1e6
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
	scheduler = get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.max_steps)

	print(f"\n=== Running {moe_spec.name} ===")
	print(f"parameters={params_m:.2f}M, active_experts={moe_spec.top_k}/{moe_spec.num_experts}, expert_hidden={moe_spec.expert_hidden_size}")
	history: list[dict[str, float | int | str]] = []
	running_loss = 0.0
	last_log_step = 0
	start_time = time.time()
	train_iter = iter(train_loader)

	model.train()
	progress = tqdm(range(1, cfg.max_steps + 1), desc=moe_spec.name, dynamic_ncols=True)
	for step in progress:
		try:
			batch = next(train_iter)
		except StopIteration:
			train_iter = iter(train_loader)
			batch = next(train_iter)

		batch = {k: v.to(device) for k, v in batch.items() if k in {"input_ids", "labels"}}
		with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
			loss = model(**batch)["loss"]

		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad(set_to_none=True)
		model.update_moe_biases()
		running_loss += float(loss.detach().cpu())
		progress.set_postfix(
			loss=f"{float(loss.detach().cpu()):.4f}",
			lr=f"{scheduler.get_last_lr()[0]:.2e}",
			bias=f"{model.moe_load_stats()['moe_bias_abs_mean']:.4f}",
		)

		if step % cfg.eval_steps == 0 or step == 1:
			steps_since_log = step - last_log_step
			train_loss = running_loss / max(1, steps_since_log)
			running_loss = 0.0
			last_log_step = step
			eval_loss = evaluate(model, eval_loader, device, autocast_dtype, cfg.max_eval_batches)
			moe_stats = model.moe_load_stats()
			elapsed = time.time() - start_time
			record = {
				"spec": moe_spec.name,
				"step": step,
				"train_loss": train_loss,
				"eval_loss": eval_loss,
				"eval_ppl": math.exp(eval_loss) if eval_loss < 20 else float("inf"),
				"lr": scheduler.get_last_lr()[0],
				"params_m": params_m,
				"moe_load_cv": moe_stats["moe_load_cv"],
				"moe_load_max_over_mean": moe_stats["moe_load_max_over_mean"],
				"moe_bias_abs_mean": moe_stats["moe_bias_abs_mean"],
				"elapsed_sec": elapsed,
			}
			history.append(record)
			progress.write(
				f"[{moe_spec.name}] step={step:5d} train_loss={train_loss:.4f} "
				f"eval_loss={eval_loss:.4f} ppl={record['eval_ppl']:.2f} lr={record['lr']:.2e} "
				f"load_cv={record['moe_load_cv']:.3f} bias_abs={record['moe_bias_abs_mean']:.4f}"
			)

		if cfg.save_steps > 0 and step % cfg.save_steps == 0:
			torch.save(model.state_dict(), run_dir / f"{moe_spec.name}_step{step}.pt")

	torch.save(model.state_dict(), run_dir / f"{moe_spec.name}_final.pt")
	return history


def write_history(history: list[dict[str, float | int | str]], output_dir: Path) -> None:
	json_path = output_dir / "history.json"
	csv_path = output_dir / "history.csv"
	json_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
	if history:
		with csv_path.open("w", newline="", encoding="utf-8") as f:
			writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
			writer.writeheader()
			writer.writerows(history)

		final_by_spec = {}
		for row in history:
			final_by_spec[row["spec"]] = row
		summary = sorted(final_by_spec.values(), key=lambda row: float(row["eval_loss"]))
		(output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
		print("\n=== Final ranking by eval_loss ===")
		for rank, row in enumerate(summary, start=1):
			print(f"#{rank} {row['spec']}: eval_loss={float(row['eval_loss']):.4f}, train_loss={float(row['train_loss']):.4f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Ablate equal-active/equal-total-param top-k MoE ranks on orca_math_qa.")
	parser.add_argument("--specs", default="A,B,C", help="Comma-separated specs to run: A,B,C")
	parser.add_argument("--dataset-path", default=AblationConfig.dataset_path)
	parser.add_argument("--tokenizer-name", default=AblationConfig.tokenizer_name)
	parser.add_argument("--output-dir", default=AblationConfig.output_dir)
	parser.add_argument("--seq-len", type=int, default=AblationConfig.seq_len)
	parser.add_argument("--batch-size", type=int, default=AblationConfig.batch_size)
	parser.add_argument("--max-steps", type=int, default=AblationConfig.max_steps)
	parser.add_argument("--eval-steps", type=int, default=AblationConfig.eval_steps)
	parser.add_argument("--learning-rate", type=float, default=AblationConfig.learning_rate)
	parser.add_argument("--warmup-steps", type=int, default=AblationConfig.warmup_steps)
	parser.add_argument("--router-bias-update-rate", type=float, default=AblationConfig.router_bias_update_rate)
	parser.add_argument("--router-score-func", choices=["sigmoid", "softmax"], default=AblationConfig.router_score_func)
	parser.add_argument("--seed", type=int, default=AblationConfig.seed)
	parser.add_argument("--max-train-samples", type=int, default=None)
	parser.add_argument("--max-eval-batches", type=int, default=AblationConfig.max_eval_batches)
	parser.add_argument("--no-bf16", action="store_true")
	parser.add_argument("--overwrite-output-dir", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = AblationConfig(
		dataset_path=args.dataset_path,
		tokenizer_name=args.tokenizer_name,
		output_dir=args.output_dir,
		seq_len=args.seq_len,
		batch_size=args.batch_size,
		max_steps=args.max_steps,
		eval_steps=args.eval_steps,
		learning_rate=args.learning_rate,
		warmup_steps=args.warmup_steps,
		router_bias_update_rate=args.router_bias_update_rate,
		router_score_func=args.router_score_func,
		seed=args.seed,
		max_train_samples=args.max_train_samples,
		max_eval_batches=args.max_eval_batches,
		bf16=not args.no_bf16,
		overwrite_output_dir=args.overwrite_output_dir,
	)
	output_dir = Path(cfg.output_dir)
	if output_dir.exists() and cfg.overwrite_output_dir:
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

	tokenizer, tokenized, collator = prepare_datasets(cfg)
	selected_specs = [spec.strip().upper() for spec in args.specs.split(",") if spec.strip()]
	history: list[dict[str, float | int | str]] = []
	for spec_key in selected_specs:
		if spec_key not in MOE_SPECS:
			raise ValueError(f"Unknown spec '{spec_key}'. Valid specs: {sorted(MOE_SPECS)}")
		history.extend(train_one_spec(cfg, MOE_SPECS[spec_key], tokenizer, tokenized, collator, output_dir))
		write_history(history, output_dir)


if __name__ == "__main__":
	main()
