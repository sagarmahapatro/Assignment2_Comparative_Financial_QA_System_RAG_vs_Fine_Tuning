from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import json
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import pickle

class RagIndex:
    def __init__(self, emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.emb_model = emb_model
        self.emb_model = SentenceTransformer(emb_model)
        self.chunks: List[Dict[str, Any]] = []
        self.faiss_index = None
        self.embeddings = None
        self.tokenized = None
        self.bm25 = None

    def build_index(self, processed_dir: Path):
        print(f"[bold]Building hybrid index with model: {self.emb_model}[/]")
        self.chunks = json.loads((processed_dir / "chunks.json").read_text(encoding="utf-8"))
        print(f"[bold]Building hybrid index on {len(self.chunks)} chunks...[/]")

        # Implement the index building logic here
        texts = [c["text"] for c in self.chunks]

        # Dense
        self.embeddings = self.emb_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

        # Sparse
        self.tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.tokenized)

        # FAISS index
        dim = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)  # cosine if normalized
        self.faiss_index.add(self.embeddings.astype(np.float32))


    def save(self, outdir: Path):
        print(f"[bold]Saving index to {outdir}[/]")

        # Save FAISS
        faiss.write_index(self.faiss_index, str(outdir / "dense.faiss"))

        # Save numpy embeddings & metadata
        np.save(outdir / "embeddings.npy", self.embeddings)
        with open(outdir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        # Save BM25
        with open(outdir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(outdir / "tokenized.pkl", "wb") as f:
            pickle.dump(self.tokenized, f)
       
       
        print("[bold]Index saved successfully![/]") 


    @classmethod
    def load(cls, index_dir: Path):
        print(f"[bold]Loading index from {index_dir}[/]")
        
        self = cls()
        self.embeddings = np.load(index_dir / "embeddings.npy")
        self.faiss_index = faiss.read_index(str(index_dir / "dense.faiss"))
      
        with open(index_dir / "chunks.pkl","rb") as f:
            self.chunks = pickle.load(f)
        with open(index_dir / "bm25.pkl","rb") as f:
            self.bm25 = pickle.load(f)
        with open(index_dir / "tokenized.pkl","rb") as f:
            self.tokenized = pickle.load(f)

        return self



      