"""Compare token compression ratios across tokenizers and datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from datasets import Value, load_dataset
from tokenizers import Tokenizer
from transformers import AutoTokenizer


DATASET_SPECS = [
    (
        "finepdfs_fineweb_mix",
        lambda: load_dataset(
            "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
            split="train",
            data_files=[f"data/train-{str(i).zfill(5)}-of-00100.parquet" for i in range(5)],
            verification_mode="no_checks",
        ).select_columns("text").cast_column("text", Value("string")),
    ),
    (
        "the_stack_smol",
        lambda: load_dataset("bigcode/the-stack-smol", split="train", verification_mode="no_checks")
        .select_columns("content")
        .rename_columns({"content": "text"})
        .cast_column("text", Value("string")),
    ),
    (
        "fineweb_edu_chinese_v2_1",
        lambda: load_dataset(
            "opencsg/Fineweb-Edu-Chinese-V2.1",
            split="train",
            data_files=[f"4_5/{str(i).zfill(6)}.parquet" for i in range(400)],
            verification_mode="no_checks",
        ).select_columns("text").cast_column("text", Value("string")),
    ),
    (
        "tiny_textbooks",
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
]


class RawTokenizersWrapper:
    def __init__(self, path: str) -> None:
        self.tokenizer = Tokenizer.from_file(path)

    def count_batch(self, texts: list[str]) -> list[int]:
        return [len(enc.ids) for enc in self.tokenizer.encode_batch(texts)]

    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


class TransformersTokenizerWrapper:
    def __init__(self, name_or_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    def count_batch(self, texts: list[str]) -> list[int]:
        encoded = self.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def vocab_size(self) -> int:
        return len(self.tokenizer)


def sample_texts(dataset, sample_size: int, seed: int) -> list[str]:
    n = min(sample_size, len(dataset))
    sampled = dataset.shuffle(seed=seed).select(range(n))
    return [text for text in sampled["text"] if isinstance(text, str) and text]


def evaluate_tokenizer(wrapper, texts: list[str], batch_size: int) -> dict[str, float]:
    token_counts: list[int] = []
    byte_counts: list[int] = []
    char_counts: list[int] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        counts = wrapper.count_batch(batch)
        token_counts.extend(counts)
        byte_counts.extend([len(text.encode("utf-8")) for text in batch])
        char_counts.extend([len(text) for text in batch])

    total_tokens = sum(token_counts)
    total_bytes = sum(byte_counts)
    total_chars = sum(char_counts)
    return {
        "samples": len(texts),
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "tokens_per_sample": mean(token_counts) if token_counts else 0.0,
        "bytes_per_token": total_bytes / total_tokens if total_tokens else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "tokens_per_1k_bytes": total_tokens / total_bytes * 1000 if total_bytes else 0.0,
        "tokens_per_1k_chars": total_tokens / total_chars * 1000 if total_chars else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/tokenizer_compression_results.json"))
    args = parser.parse_args()

    tokenizers = {
        "tokenizer_output": RawTokenizersWrapper("tokenizer_output/tokenizer.json"),
        "tokenizer_workspace": TransformersTokenizerWrapper("tokenizer_workspace"),
        "gpt2": TransformersTokenizerWrapper("gpt2"),
        "qwen3_8b": TransformersTokenizerWrapper("Qwen/Qwen3-8B"),
    }

    results = {
        "sample_size_per_dataset": args.sample_size,
        "tokenizers": {name: {"vocab_size": wrapper.vocab_size()} for name, wrapper in tokenizers.items()},
        "datasets": {},
    }

    for dataset_name, loader in DATASET_SPECS:
        print(f"\n=== {dataset_name} ===")
        dataset = loader()
        texts = sample_texts(dataset, args.sample_size, args.seed)
        dataset_results = {}
        for tokenizer_name, wrapper in tokenizers.items():
            metrics = evaluate_tokenizer(wrapper, texts, args.batch_size)
            dataset_results[tokenizer_name] = metrics
            print(
                f"{tokenizer_name:20s} "
                f"tok/sample={metrics['tokens_per_sample']:.1f} "
                f"bytes/tok={metrics['bytes_per_token']:.3f} "
                f"tok/1kB={metrics['tokens_per_1k_bytes']:.1f}"
            )
        results["datasets"][dataset_name] = dataset_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
