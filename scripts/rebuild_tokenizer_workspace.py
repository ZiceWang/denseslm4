"""Rebuild tokenizer_workspace as a real BPE tokenizer.

Procedure:
1. Sample text from the DenseSLM4 training mixture.
2. Tokenize the sample with Qwen3-8B tokenizer and keep the most frequent N tokens.
3. Keep all Qwen3 BPE merges needed to build those frequent tokens.
4. Append tokenizer_output BPE vocab/merges when compatible.
5. Save a merged BPE tokenizer into tokenizer_workspace.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from datasets import Value, concatenate_datasets, load_dataset
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|unk|>"]
DEFAULT_QWEN_TOKENIZER = "Qwen/Qwen3-8B"


def load_text_sample(sample_size: int, seed: int) -> list[str]:
    datasets = []

    dataset_specs = [
        (
            "finepdfs/fineweb mix",
            lambda: load_dataset(
                "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
                split="train",
                data_files=[f"data/train-{str(i).zfill(5)}-of-00100.parquet" for i in range(5)],
                verification_mode="no_checks",
            ).select_columns("text").cast_column("text", Value("string")),
        ),
        (
            "the-stack-smol",
            lambda: load_dataset("bigcode/the-stack-smol", split="train", verification_mode="no_checks")
            .select_columns("content")
            .rename_columns({"content": "text"})
            .cast_column("text", Value("string")),
        ),
        (
            "Fineweb-Edu-Chinese-V2.1",
            lambda: load_dataset(
                "opencsg/Fineweb-Edu-Chinese-V2.1",
                split="train",
                data_files=[f"4_5/{str(i).zfill(6)}.parquet" for i in range(400)],
                verification_mode="no_checks",
            ).select_columns("text").cast_column("text", Value("string")),
        ),
        (
            "tiny-textbooks",
            lambda: load_dataset("nampdn-ai/tiny-textbooks", split="train", verification_mode="no_checks")
            .select_columns("textbook")
            .rename_columns({"textbook": "text"})
            .cast_column("text", Value("string")),
        ),
        (
            "orca_math_qa",
            lambda: load_dataset(
                "parquet",
                split="train",
                data_files="./dataset/orca_math_qa.parquet",
                verification_mode="no_checks",
            ).select_columns("text").cast_column("text", Value("string")),
        ),
        (
            "extra local parquet",
            lambda: load_dataset(
                "parquet",
                split="train",
                data_files=["./dataset/1.parquet", "./dataset/2.parquet", "./dataset/3.parquet"],
                verification_mode="no_checks",
            ).select_columns("text").cast_column("text", Value("string")),
        ),
    ]

    for name, loader in dataset_specs:
        try:
            dataset = loader()
            if len(dataset) > 0:
                datasets.append(dataset)
            print(f"Loaded {name}: {len(dataset)} samples")
        except Exception as exc:
            print(f"Skipped {name}: {exc}")

    if not datasets:
        raise RuntimeError("No datasets could be loaded for sampling.")

    merged = concatenate_datasets(datasets).shuffle(seed=seed)
    n = min(sample_size, len(merged))
    sampled = merged.select(range(n))
    texts = [text for text in sampled["text"] if isinstance(text, str) and text]
    print(f"Sampled {len(texts)} non-empty texts from {len(merged)} total examples")
    return texts


def count_qwen_tokens(texts: list[str], qwen_tokenizer_name: str, top_n: int, batch_size: int) -> tuple[list[str], Tokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_name, trust_remote_code=True)
    counter: Counter[int] = Counter()

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False, truncation=False)["input_ids"]
        for ids in encoded:
            counter.update(ids)
        if (start // batch_size + 1) % 20 == 0:
            print(f"Counted Qwen tokens for {min(start + batch_size, len(texts))}/{len(texts)} samples")

    special_ids = set(tokenizer.all_special_ids)
    qwen_tokens: list[str] = []
    for token_id, _ in counter.most_common():
        if token_id in special_ids:
            continue
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token and token not in SPECIAL_TOKENS:
            qwen_tokens.append(token)
        if len(qwen_tokens) >= top_n:
            break

    print(f"Selected {len(qwen_tokens)} frequent Qwen tokens")
    return qwen_tokens, tokenizer.backend_tokenizer


def load_tokenizer_json(tokenizer_dir: Path) -> dict:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer.json in {tokenizer_dir}")
    return json.loads(tokenizer_path.read_text(encoding="utf-8"))


def merge_bpe_tokenizers(qwen_tokenizer: Tokenizer, qwen_tokens: Iterable[str], tokenizer_output: Path) -> Tokenizer:
    qwen_json = json.loads(qwen_tokenizer.to_str())
    local_json = load_tokenizer_json(tokenizer_output)

    if qwen_json["model"]["type"] != "BPE" or local_json["model"]["type"] != "BPE":
        raise ValueError("Both tokenizers must be BPE tokenizers.")

    base_vocab: dict[str, int] = qwen_json["model"]["vocab"]
    local_vocab: dict[str, int] = local_json["model"]["vocab"]
    qwen_merges: list[list[str]] = qwen_json["model"].get("merges", [])
    local_merges: list[list[str]] = local_json["model"].get("merges", [])

    required_tokens = set(SPECIAL_TOKENS)
    required_tokens.update(qwen_tokens)
    required_tokens.update(local_vocab)

    qwen_merge_by_output: dict[str, tuple[str, str]] = {}
    qwen_merge_rank: dict[tuple[str, str], int] = {}
    for rank, (left, right) in enumerate(qwen_merges):
        pair = (left, right)
        qwen_merge_by_output.setdefault(left + right, pair)
        qwen_merge_rank.setdefault(pair, rank)

    required_qwen_merges: set[tuple[str, str]] = set()

    def require_qwen_token(token: str) -> None:
        pair = qwen_merge_by_output.get(token)
        if pair is None:
            return
        if pair in required_qwen_merges:
            return
        left, right = pair
        required_tokens.add(left)
        required_tokens.add(right)
        required_qwen_merges.add(pair)
        require_qwen_token(left)
        require_qwen_token(right)

    for token in list(required_tokens):
        require_qwen_token(token)

    merged_vocab: dict[str, int] = {}
    for token, token_id in sorted(base_vocab.items(), key=lambda kv: kv[1]):
        if token in required_tokens:
            merged_vocab[token] = len(merged_vocab)

    merged_merges: list[tuple[str, str]] = []
    seen_merges: set[tuple[str, str]] = set()

    for left, right in sorted(required_qwen_merges, key=lambda pair: qwen_merge_rank[pair]):
        out = left + right
        pair = (left, right)
        if left in merged_vocab and right in merged_vocab:
            if pair not in seen_merges:
                merged_merges.append(pair)
                seen_merges.add(pair)
            if out not in merged_vocab:
                merged_vocab[out] = len(merged_vocab)

    for token, _ in sorted(local_vocab.items(), key=lambda kv: kv[1]):
        if token not in merged_vocab:
            merged_vocab[token] = len(merged_vocab)

    for merge in local_merges:
        left, right = merge
        pair = (left, right)
        out = left + right
        if left in merged_vocab and right in merged_vocab and out in merged_vocab and pair not in seen_merges:
            merged_merges.append(pair)
            seen_merges.add(pair)

    model_cfg = qwen_json["model"]
    tokenizer = Tokenizer(
        models.BPE(
            vocab=merged_vocab,
            merges=merged_merges,
            unk_token=model_cfg.get("unk_token"),
            fuse_unk=model_cfg.get("fuse_unk", False),
            byte_fallback=model_cfg.get("byte_fallback", False),
            ignore_merges=model_cfg.get("ignore_merges", False),
        )
    )
    tokenizer.normalizer = qwen_tokenizer.normalizer
    tokenizer.pre_tokenizer = qwen_tokenizer.pre_tokenizer
    tokenizer.decoder = qwen_tokenizer.decoder
    tokenizer.post_processor = qwen_tokenizer.post_processor

    print(f"Merged BPE vocab size: {len(merged_vocab)}")
    print(f"Merged BPE merges: {len(merged_merges)}")
    return tokenizer


def load_bpe_tokens(tokenizer_output: Path) -> list[str]:
    vocab_path = tokenizer_output / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        return [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]

    tokenizer_path = tokenizer_output / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    return [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def merge_tokens(qwen_tokens: Iterable[str], bpe_tokens: Iterable[str]) -> dict[str, int]:
    merged: list[str] = []
    seen: set[str] = set()

    for token in SPECIAL_TOKENS:
        if token not in seen:
            seen.add(token)
            merged.append(token)

    for source in (qwen_tokens, bpe_tokens):
        for token in source:
            if token and token not in seen:
                seen.add(token)
                merged.append(token)

    return {token: idx for idx, token in enumerate(merged)}


def save_tokenizer(tokenizer: Tokenizer, output_dir: Path) -> None:
    if output_dir.exists():
        backup_dir = output_dir.with_name(output_dir.name + ".bak")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(output_dir, backup_dir)
        print(f"Backed up existing tokenizer workspace to {backup_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(str(output_dir / "tokenizer.json"))

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(output_dir / "tokenizer.json"),
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        unk_token="<|unk|>",
    )
    fast_tokenizer.save_pretrained(output_dir)

    print(f"Saved merged tokenizer to {output_dir}")
    print(f"Final vocab size: {tokenizer.get_vocab_size()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-tokenizer", default=DEFAULT_QWEN_TOKENIZER)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--top-qwen-tokens", type=int, default=19000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-output", type=Path, default=Path("tokenizer_output"))
    parser.add_argument("--output-dir", type=Path, default=Path("tokenizer_workspace"))
    args = parser.parse_args()

    texts = load_text_sample(args.sample_size, args.seed)
    qwen_tokens, qwen_tokenizer = count_qwen_tokens(texts, args.qwen_tokenizer, args.top_qwen_tokens, args.batch_size)
    bpe_tokens = load_bpe_tokens(args.tokenizer_output)
    print(f"Loaded {len(bpe_tokens)} BPE tokens from {args.tokenizer_output}")
    tokenizer = merge_bpe_tokenizers(qwen_tokenizer, qwen_tokens, args.tokenizer_output)
    save_tokenizer(tokenizer, args.output_dir)


if __name__ == "__main__":
    main()
