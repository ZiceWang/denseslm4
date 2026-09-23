# SSLM — A Super Small but Usable Language Model at GPT-2 Scale

`denseslm4` is the training and evaluation codebase for **SSLM**, a hybrid decoder-only language
model aiming for *useful agent-level capabilities at GPT-2 scale*. SSLM v2 carries **1.45B total
parameters** but only **~768M active parameters per forward pass**.

Paper: [`papers/main.tex`](papers/main.tex) / [`papers/main.pdf`](papers/main.pdf) —
*SSLM: Achieving Useful Agent Capabilities at GPT-2 Scale* (Zice Wang, NTAIRS, Northeastern University).

## Architecture

SSLM v2 combines three ingredients:

- **Multi-head Latent Attention (MLA)** — compressed KV cache (4 KV heads, 512-dim) for
  memory-efficient multi-turn context.
- **Mamba2 state-space layers** — efficient long-context processing.
- **Aux-loss-free Mixture-of-Experts (MoE)** — 64 experts, top-2 routing; parameter-efficient
  scaling without load-balancing auxiliary losses.

| Property | GPT-2 XL | SSLM v2 |
|---|---|---|
| Architecture | dense Transformer | hybrid MLA + Mamba2 + MoE |
| Attention | full MHA (16 heads) | MLA compressed (4 KV heads, 512-dim) |
| FFN | dense | sparse MoE (64 experts, top-2) |
| KV cache | full, O(N) growth | compressed, constant 512-dim/head |
| Layers | 48 | 48 |
| Hidden size | 1600 | 768 |
| Vocab size | 50,257 | 52,877 |
| Active params / forward | ~1.5B | **~768M** |
| Total params | 1.5B | 1.45B |

## Models and training runs

| Run | Model | Scale | Steps | Loss | PPL | Status |
|---|---|---|---|---|---|---|
| `tinystories` | DenseSLM4 (dense) | 7M | 1,559 | 3.050 | 21.11 | completed (smoke test) |
| `my_model_new_tokenizer` | DenseSLM4 (dense) | 105M | 122,837 | 2.143 | 8.52 | completed |
| `denseslm4_moe` | DenseSLM4MoE v1 | ~1.4B | 129,107 | 2.622 | 13.76 | completed |
| `denseslm4_moe_continued` | DenseSLM4MoE v1 + continued | ~1.4B | +19,367 | 2.591 | 13.34 | completed |
| `denseslm4_moe_v2` | DenseSLM4MoE v2 | **1.45B** | 190,967 | 2.441 | 11.49 | completed |

## Evaluation

LightEval results (raw JSON under [`lighteval_results/`](lighteval_results),
[`lighteval_results_moe/`](lighteval_results_moe), [`lighteval_results_moe_v2/`](lighteval_results_moe_v2)):

| Model | Benchmark | Metric | Score |
|---|---|---|---|
| DenseSLM4MoE v1 (1.4B, 22K vocab) | C-Eval (avg of 43 subjects) | accuracy, 0-shot / 5-shot | 0.217 / **0.281** |
| DenseSLM4MoE v1 | TruthfulQA | MC1 / MC2 | 0.203 / 0.440 |
| DenseSLM4 (dense, 105M) | GSM8K (5-shot) | extractive match | 0.076% |
| DenseSLM4 (dense, 105M) | TruthfulQA | MC1 / MC2 | 0.240 / 0.430 |

Note (paper §7): these numbers are modest for models at this scale without extensive RLHF; the
reported gap analysis targets stronger Chinese reasoning, truthfulness, and math after more data,
CoT, and RLHF. v2 evaluation is pending the completion of training.

## Repository layout

```text
denseslm4/
├── src/denseslm4/            # model + training package
│   ├── configuration_denseslm4.py / _moe.py   # HF-style configs
│   ├── modeling_denseslm4.py / _moe.py        # hybrid MLA + Mamba2 + MoE model
│   ├── layers/               # mla.py, mamba2.py, deltaformer.py, latent_deltaformer.py
│   ├── modules.py            # shared blocks
│   ├── muon.py               # Muon optimizer
│   ├── tilelang_causal_conv1d.py + test_tilelang.py  # TileLang causal-conv1d kernel
│   ├── train.py / train_moe.py                # pretraining entry points (Typer CLIs)
│   ├── sft_moe.py            # supervised fine-tuning
│   └── pretrained_dataset.py # streaming pretraining dataset
├── scripts/                  # tokenizer training, chat template, generation & LightEval tests
├── papers/                   # LaTeX source + PDF of the SSLM report
├── lighteval_results*/       # LightEval outputs (C-Eval / TruthfulQA / GSM8K)
├── benchmark_overlap.py, benchmark_tilelang_causal_conv1d.py   # kernel benchmarks
├── notebooks/
├── tokenizer_output/, tokenizer_workspace/   # trained tokenizer artifacts
└── pyproject.toml            # uv-managed deps + `train` / `train-moe` entry points
```

`runs/` (checkpoints), `dataset/`, and `.venv/` are gitignored.

## Quickstart

```bash
uv venv && uv sync            # or: uv pip install -e .

# Typer CLIs
python -m denseslm4.train --help
python -m denseslm4.train_moe --help

# installed console scripts
train --help
train-moe --help
```

Core dependencies: `torch>=2.11` (cu128), `transformers>=5.8`, `flash-attn`, `mamba-ssm`,
`causal-conv1d`, `flash-linear-attention`, `tilelang`, `trl`, `lighteval`, `lightning`.

## Status

- `DenseSLM4` (dense 105M) and `DenseSLM4MoE` v1 (~1.4B) are trained and evaluated.
- `DenseSLM4MoE` v2 (1.45B) pretraining is complete: one epoch over ~6.3B tokens
  (190,967 steps), evaluation perplexity 11.49; a preliminary SFT stage reaches 8.34.
- Next steps (scaling compute, matched baselines, post-training, long context, ablations)
  are listed in `papers/sections/9_future_work.tex`.
