"""预训练数据集加载模块。直接调用 `load_pretrained_dataset()` 即可获取合并后的数据集。"""

from datasets import load_dataset, concatenate_datasets, Dataset
from datasets import Value


def load_pretrained_dataset() -> tuple[Dataset, str]:
    """
    加载并合并所有预训练数据集。
    
    Returns:
        tuple[Dataset, str]: (FINAL_DATASET, TEXT_COLUMN)
            - FINAL_DATASET: 合并后的完整数据集
            - TEXT_COLUMN: 文本列名 ("text")
    """
    print("Start download/validate the datasets.")
    # current speed is about 502288 samples/h

    sample_cnt = 0
    # ------------------- 加载第一个数据集 -------------------
    data_files_1 = [
        f"data/train-{str(i).zfill(5)}-of-00100.parquet"
        for i in range(5)  # 10 GB for 5 files, about 5 hours of training
    ]

    dataset_1 = load_dataset(
        "HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled",
        split="train",
        data_files=data_files_1,
        verification_mode="no_checks"
    )
    dataset_1 = dataset_1.select_columns("text").cast_column("text", Value("string"))
    print(f"Dataset 1 loaded with {len(dataset_1)} samples.")
    sample_cnt += len(dataset_1)

    # ------------------- 加载第二个数据集 -------------------
    dataset_2 = load_dataset(
        "bigcode/the-stack-smol",
        split="train",  # 2GB 300000 samples, about 0.6 hour of training
    )
    dataset_2 = dataset_2.select_columns("content").rename_columns({"content": "text"}).cast_column("text", Value("string"))
    print(f"Dataset 2 loaded with {len(dataset_2)} samples.")
    sample_cnt += len(dataset_2)

    # ------------------- 加载第三个数据集 -------------------
    data_files_3 = [
        f"4_5/{str(i).zfill(6)}.parquet"
        for i in range(400)  # 4 GB for 400 about 2 hours of training
    ]

    dataset_3 = load_dataset(
        "opencsg/Fineweb-Edu-Chinese-V2.1",
        split="train",
        data_files=data_files_3,
        verification_mode="no_checks",
    )
    dataset_3 = dataset_3.select_columns("text").cast_column("text", Value("string"))
    print(f"Dataset 3 loaded with {len(dataset_3)} samples.")
    sample_cnt += len(dataset_3)

    dataset_4 = load_dataset("nampdn-ai/tiny-textbooks", split="train")  # ~ 1B 420,000 samples
    dataset_4 = dataset_4.select_columns("textbook").rename_columns({"textbook": "text"}).cast_column("text", Value("string"))

    print(f"Dataset 4 loaded with {len(dataset_4)} samples.")
    sample_cnt += len(dataset_4)

    # ==================== 加载第五个数据集 ====================
    dataset_5 = load_dataset("parquet", split="train", data_files="./dataset/orca_math_qa.parquet", verification_mode="no_checks")
    dataset_5 = dataset_5.select_columns("text").cast_column("text", Value("string"))
    print(f"Dataset 5 loaded with {len(dataset_5)} samples.")
    sample_cnt += len(dataset_5)
    # ==================== 核心补充：加载额外3个parquet并按0.4%采样 ====================
    print(f"\n总基础样本数: {sample_cnt}，0.4% 采样数量: {int(0.004 * sample_cnt)}")
    spice_sample = int(0.004 * sample_cnt)
    
    # 定义额外数据集路径（你的本地路径）
    extra_data_files = [
        "./dataset/1.parquet",
        "./dataset/2.parquet",
        "./dataset/3.parquet"
    ]

    # 加载额外的3个parquet数据集
    extra_dataset = load_dataset(
        "parquet",
        split="train",
        data_files=extra_data_files,
        verification_mode="no_checks"
    )

    extra_dataset = extra_dataset.select_columns("text").cast_column("text", Value("string"))
    print(f"额外数据集总样本数: {len(extra_dataset)}")

    # 随机采样 0.4% 比例的样本
    # 若额外数据集样本不足，则取全部样本
    if len(extra_dataset) >= spice_sample:
        extra_dataset_sampled = extra_dataset.shuffle(seed=42).select(range(spice_sample))
    else:
        extra_dataset_sampled = extra_dataset
        print(f"警告：额外数据集样本不足，使用全部 {len(extra_dataset)} 个样本")

    print(f"采样后额外数据集样本数: {len(extra_dataset_sampled)}")

    # ==================== 合并所有数据集 ====================
    FINAL_DATASET = concatenate_datasets([
        dataset_1,
        dataset_2,
        dataset_3,
        dataset_4,
        dataset_5,
        extra_dataset_sampled
    ])
    TEXT_COLUMN = "text"

    # ==================== 最终统计 ====================
    print("\n" + "="*50)
    print(f"✅ 所有数据集加载完成！")
    print(f"基础数据集总样本数: {sample_cnt}")
    print(f"额外采样样本数: {len(extra_dataset_sampled)}")
    print(f"最终合并总样本数: {len(FINAL_DATASET)}")
    print(f"预计训练时间 (基于 502288 样本/小时): {len(FINAL_DATASET) / 502288:.2f} 小时")
    print("="*50)

    return FINAL_DATASET, TEXT_COLUMN

if __name__ == "__main__":
    # 模块加载时自动执行并导出
    final_dataset, text_column = load_pretrained_dataset()