"""Train a BPE tokenizer on the DenseSLM4 dataset and save vocabulary."""

from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors
from datasets import load_dataset, concatenate_datasets, Dataset
from datasets import Value
import random


def load_sample_dataset(sample_size: int = 500000, seed: int = 42) -> Dataset:
    """Load a sample of the training dataset for tokenizer training."""
    print(f"Loading dataset sample (target: {sample_size} samples)...")

    sample_cnt = 0
    texts = []

    # Dataset 1: fineweb-edu
    data_files_1 = [
        f"data/train-{str(i).zfill(5)}-of-00100.parquet"
        for i in range(5)
    ]
    try:
        dataset_1 = load_dataset(
            "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
            split="train",
            data_files=data_files_1,
            verification_mode="no_checks"
        )
        dataset_1 = dataset_1.select_columns("text").cast_column("text", Value("string"))
        sample_cnt += len(dataset_1)
        print(f"Dataset 1 loaded: {len(dataset_1)} samples")
    except Exception as e:
        print(f"Dataset 1 failed: {e}")

    # Dataset 2: the-stack-smol
    try:
        dataset_2 = load_dataset(
            "bigcode/the-stack-smol",
            split="train",
            verification_mode="no_checks"
        )
        dataset_2 = dataset_2.select_columns("content").rename_columns({"content": "text"}).cast_column("text", Value("string"))
        sample_cnt += len(dataset_2)
        print(f"Dataset 2 loaded: {len(dataset_2)} samples")
    except Exception as e:
        print(f"Dataset 2 failed: {e}")

    # Dataset 3: Fineweb-Edu-Chinese-V2.1
    data_files_3 = [
        f"4_5/{str(i).zfill(6)}.parquet"
        for i in range(400)
    ]
    try:
        dataset_3 = load_dataset(
            "opencsg/Fineweb-Edu-Chinese-V2.1",
            split="train",
            data_files=data_files_3,
            verification_mode="no_checks",
        )
        dataset_3 = dataset_3.select_columns("text").cast_column("text", Value("string"))
        sample_cnt += len(dataset_3)
        print(f"Dataset 3 loaded: {len(dataset_3)} samples")
    except Exception as e:
        print(f"Dataset 3 failed: {e}")

    # Dataset 4: tiny-textbooks
    try:
        dataset_4 = load_dataset("nampdn-ai/tiny-textbooks", split="train", verification_mode="no_checks")
        dataset_4 = dataset_4.select_columns("textbook").rename_columns({"textbook": "text"}).cast_column("text", Value("string"))
        sample_cnt += len(dataset_4)
        print(f"Dataset 4 loaded: {len(dataset_4)} samples")
    except Exception as e:
        print(f"Dataset 4 failed: {e}")

    # Extra local datasets
    extra_data_files = [
        "/data1/neu_lab2/denseslm4/dataset/1.parquet",
        "/data1/neu_lab2/denseslm4/dataset/2.parquet",
        "/data1/neu_lab2/denseslm4/dataset/3.parquet"
    ]
    try:
        extra_dataset = load_dataset(
            "parquet",
            split="train",
            data_files=extra_data_files,
            verification_mode="no_checks"
        )
        extra_dataset = extra_dataset.select_columns("text").cast_column("text", Value("string"))
        sample_cnt += len(extra_dataset)
        print(f"Extra dataset loaded: {len(extra_dataset)} samples")
    except Exception as e:
        print(f"Extra dataset failed: {e}")

    print(f"\nTotal available samples: {sample_cnt}")

    # Sample from each dataset proportionally
    all_texts = []
    for ds in [dataset_1, dataset_2, dataset_3, dataset_4]:
        try:
            if len(ds) > 0:
                ds_size = len(ds)
                n_sample = max(1, int(sample_size * ds_size / sample_cnt))
                n_sample = min(n_sample, len(ds))
                sampled = ds.shuffle(seed=seed).select(range(n_sample))
                all_texts.extend([t for t in sampled["text"] if t])
        except NameError:
            pass

    try:
        if len(extra_dataset) > 0:
            n_sample = max(1, int(sample_size * len(extra_dataset) / sample_cnt))
            n_sample = min(n_sample, len(extra_dataset))
            sampled = extra_dataset.shuffle(seed=seed).select(range(n_sample))
            all_texts.extend([t for t in sampled["text"] if t])
    except NameError:
        pass

    print(f"Collected {len(all_texts)} text samples for tokenizer training")
    return all_texts


def train_bpe_tokenizer(texts, vocab_size: int = 10000, save_dir: str = "tokenizer_output"):
    """Train a BPE tokenizer on the given texts."""
    print(f"\nTraining BPE tokenizer with target vocab_size={vocab_size}...")

    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Use ByteLevel pre-tokenizer (handles any language well)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Define special tokens (from Qwen3)
    special_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

    # Trainer with BPE
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
    )

    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add decoder for ByteLevel
    tokenizer.decoder = decoders.ByteLevel()

    # Add post-processor for special tokens (like Qwen3 chat template)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Save
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path / "tokenizer.json"))

    # Also save vocab and merges separately for inspection
    tokenizer.model.save(str(save_path))

    print(f"\nTokenizer saved to {save_path}/")

    # Print stats
    vocab = tokenizer.get_vocab()
    print(f"Final vocab size: {len(vocab)}")
    print(f"Special tokens: {special_tokens}")

    return tokenizer, save_path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--save-dir", type=str, default="tokenizer_output")
    args = parser.parse_args()

    texts = load_sample_dataset(sample_size=args.sample_size)
    tokenizer, save_path = train_bpe_tokenizer(texts, args.vocab_size, args.save_dir)

    print(f"\nDone! Tokenizer files saved to: {save_path}")


if __name__ == "__main__":
    main()