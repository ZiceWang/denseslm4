from datasets import load_dataset, concatenate_datasets, Value
# {'question':'xxx', 'answer':'xxx'} -> {'text': 'Question: xxx \n Answer: xxx'}

# microsoft/orca-math-word-problems-200k

def convert_qa_to_text(dataset):
    def convert(example):
        question = example['question']
        answer = example['answer']
        text = f"Question: {question} \n Answer: {answer}"
        return {'text': text}
    
    return dataset.map(convert, num_proc=4)

# Load dataset and convert
dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
dataset = convert_qa_to_text(dataset)
# only keep 'text' column and ensure it's string type
dataset = dataset.select_columns("text").cast_column("text", Value("string"))

# Save to parquet
dataset.to_parquet("dataset/orca_math_qa.parquet")