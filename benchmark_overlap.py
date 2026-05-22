"""
Overlap / Sliding Window preprocessing benchmark.
Tests tokenization + chunking speed on real datasets.
"""

import time
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# Config
BLOCK_SIZE = 1024
OVERLAP = 128
STRIDE = BLOCK_SIZE - OVERLAP  # 896
TOKENIZER_PATH = "tokenizer_workspace"

print("=" * 70)
print("Overlap Preprocessing Speed Benchmark")
print(f"block_size={BLOCK_SIZE}, overlap={OVERLAP}, stride={STRIDE}")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def sliding_window_chunks(token_ids, block_size=1024, stride=896):
    """Cut token_ids into overlapping chunks."""
    if len(token_ids) <= block_size:
        return [token_ids]
    chunks = []
    for i in range(0, len(token_ids) - block_size + 1, stride):
        chunks.append(token_ids[i:i + block_size])
    # Last chunk: if remaining > 0, pad or keep partial?
    # For pretraining, we usually drop the last partial chunk
    # But let's also capture tail if it has meaningful length
    remaining = len(token_ids) - len(chunks) * stride
    if remaining > block_size // 4:  # > 256 tokens, worth keeping
        chunks.append(token_ids[-block_size:])
    return chunks


def process_batch(texts, tokenizer, block_size=1024, stride=896):
    """Process a batch of texts with overlap."""
    all_chunks = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = sliding_window_chunks(token_ids, block_size, stride)
        all_chunks.extend(chunks)
    return all_chunks


# ========== Benchmark 1: finepdfs (1 file, ~500k samples) ==========
print("\n[Benchmark 1] finepdfs_edu (1 parquet file)")
ds = load_dataset(
    "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
    split="train",
    data_files=["data/train-00000-of-00100.parquet"],
    verification_mode="no_checks"
)
print(f"  Samples: {len(ds)}")

# Sample 1000 for benchmark
sample_size = min(1000, len(ds))
texts = [t['text'] for t in ds.select(range(sample_size))]

# Baseline: truncation (current method)
t0 = time.time()
baseline_outputs = []
for text in texts:
    baseline_outputs.append(tokenizer.encode(text, truncation=True, max_length=BLOCK_SIZE, add_special_tokens=False))
t1 = time.time()
baseline_time = t1 - t0
print(f"  Baseline (truncate): {baseline_time:.2f}s for {sample_size} samples")
print(f"  Throughput: {sample_size / baseline_time:.0f} samples/s")

# Overlap method
t0 = time.time()
overlap_outputs = []
for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = sliding_window_chunks(token_ids, BLOCK_SIZE, STRIDE)
    overlap_outputs.extend(chunks)
t1 = time.time()
overlap_time = t1 - t0
print(f"  Overlap (stride={STRIDE}): {overlap_time:.2f}s for {sample_size} samples")
print(f"  Throughput: {sample_size / overlap_time:.0f} samples/s")
print(f"  Output samples: {len(baseline_outputs)} → {len(overlap_outputs)} (x{len(overlap_outputs)/len(baseline_outputs):.2f})")
print(f"  Overhead: {overlap_time/baseline_time:.1f}x slower than baseline")

# Estimate full processing time
print(f"\n  Estimated full file processing: {overlap_time / sample_size * len(ds) / 60:.1f} min")


# ========== Benchmark 2: the-stack-smol (sample 1000) ==========
print("\n[Benchmark 2] the-stack-smol")
ds2 = load_dataset("bigcode/the-stack-smol", split="train")
sample_size = 1000
texts = [t['content'] for t in ds2.select(range(sample_size))]

t0 = time.time()
overlap_outputs = []
for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = sliding_window_chunks(token_ids, BLOCK_SIZE, STRIDE)
    overlap_outputs.extend(chunks)
t1 = time.time()
overlap_time = t1 - t0
print(f"  Samples: {sample_size}")
print(f"  Overlap time: {overlap_time:.2f}s")
print(f"  Output samples: {len(overlap_outputs)} (x{len(overlap_outputs)/sample_size:.2f})")
print(f"  Throughput: {sample_size / overlap_time:.0f} raw samples/s")
print(f"  Chunk throughput: {len(overlap_outputs) / overlap_time:.0f} chunks/s")


# ========== Benchmark 3: Fineweb-Chinese (1 file) ==========
print("\n[Benchmark 3] Fineweb-Edu-Chinese (1 parquet)")
ds3 = load_dataset(
    "opencsg/Fineweb-Edu-Chinese-V2.1",
    split="train",
    data_files=["4_5/000000.parquet"],
    verification_mode="no_checks"
)
sample_size = min(1000, len(ds3))
texts = [t['text'] for t in ds3.select(range(sample_size))]

t0 = time.time()
overlap_outputs = []
for text in texts:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = sliding_window_chunks(token_ids, BLOCK_SIZE, STRIDE)
    overlap_outputs.extend(chunks)
t1 = time.time()
overlap_time = t1 - t0
print(f"  Samples: {sample_size}")
print(f"  Overlap time: {overlap_time:.2f}s")
print(f"  Output samples: {len(overlap_outputs)} (x{len(overlap_outputs)/sample_size:.2f})")
print(f"  Throughput: {sample_size / overlap_time:.0f} raw samples/s")
print(f"  Chunk throughput: {len(overlap_outputs) / overlap_time:.0f} chunks/s")


# ========== Benchmark 4: Batched tokenization speed ==========
print("\n[Benchmark 4] Batched tokenization (amortize overhead)")
BATCH_SIZE = 1000
texts_batch = [t['text'] for t in ds.select(range(BATCH_SIZE))]

t0 = time.time()
# Fast batched tokenization
encoded = tokenizer(texts_batch, add_special_tokens=False, padding=False, truncation=False)
t1 = time.time()
print(f"  Batched encode {BATCH_SIZE} samples: {t1-t0:.2f}s")
print(f"  Batched throughput: {BATCH_SIZE/(t1-t0):.0f} samples/s")

# Then chunk
t0 = time.time()
all_chunks = []
for ids in encoded['input_ids']:
    chunks = sliding_window_chunks(ids, BLOCK_SIZE, STRIDE)
    all_chunks.extend(chunks)
t1 = time.time()
print(f"  Chunking time: {t1-t0:.2f}s")
print(f"  Total output: {len(all_chunks)} chunks")
print(f"  Total throughput: {BATCH_SIZE / (t1-t0 + 0.01):.0f} raw samples/s (encode+chunk)")


# ========== Summary ==========
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

# Calculate expected total data size with overlap=128
finepdfs_raw = 500458 * 5
stack_raw = 300000
fineweb_raw = 1772 * 400
textbooks_raw = 399000
orca_raw = 200035

# Expansion ratios from earlier analysis
ratios = {
    'finepdfs': 2.22,
    'stack-smol': 2.92,
    'fineweb-ch': 1.14,
    'textbooks': 1.00,
    'orca': 1.00,
}

current_total = finepdfs_raw + stack_raw + fineweb_raw + textbooks_raw + orca_raw
new_total = (finepdfs_raw * ratios['finepdfs'] + 
             stack_raw * ratios['stack-smol'] + 
             fineweb_raw * ratios['fineweb-ch'] + 
             textbooks_raw * ratios['textbooks'] + 
             orca_raw * ratios['orca'])

print(f"\nCurrent total samples: {current_total:,.0f}")
print(f"With overlap=128:      {new_total:,.0f}")
print(f"Expansion:             {new_total/current_total:.2f}x")

# Steps at batch_size=32
current_steps = current_total / 32
new_steps = new_total / 32
print(f"\nCurrent steps (batch=32): {current_steps:,.0f}")
print(f"New steps:                {new_steps:,.0f}")
print(f"Extra steps:              {new_steps - current_steps:,.0f}")

# Time estimate at current training speed
# Current: 129107 steps in ~6.4 hours (with restarts), effective maybe ~4-5 hours
# Let's use 502288 samples/hour as stated in code
samples_per_hour = 502288
print(f"\nAt {samples_per_hour:,} samples/hour:")
print(f"  Current 1 epoch: {current_total/samples_per_hour:.1f} hours")
print(f"  New 1 epoch:     {new_total/samples_per_hour:.1f} hours")
print(f"  Difference:      +{(new_total-current_total)/samples_per_hour:.1f} hours")
