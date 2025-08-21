
from pathlib import Path
import pandas as pd
from rank_bm25 import BM25Okapi

class MemoryBank:
    def __init__(self, path: Path):
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame(columns=["question","answer"])
        self.df = df
        self.tokenized = [str(q).lower().split() for q in df["question"].tolist()]
        self.bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def query(self, q: str, top_k: int = 3):
        if self.bm25 is None:
            return []
        toks = q.lower().split()
        scores = self.bm25.get_scores(toks)
        best_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.df.iloc[i].to_dict() for i in best_idx]