from datasets import Dataset
import pandas as pd

def load_qa_data_for_finetune(qa_path) -> str:
    df = pd.read_csv(qa_path)
    df["text"] = "Question: " + df["question"].astype(str) + "\nAnswer: " + df["answer"].astype(str)
    dset = Dataset.from_pandas(df[["text"]])
    return dset