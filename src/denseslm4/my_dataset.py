import torch
from datasets import load_dataset
data_files = [
    f"data/train-{str(i).zfill(5)}-of-00100.parquet" 
    for i in range(10)  # 0到59，共60个文件
]

dataset = load_dataset(
    "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
    split="train",
    data_files=data_files,  # 关键参数：指定只加载这些文件
    verification_mode="no_checks",  # 禁用大小校验
)
