from typing import List, Dict, Any, Tuple
from ..rag.index import RagIndex
from ..utils.text_cleaner import lower_strip_punct
import numpy as np 

def hybrid_search(query: str, index: RagIndex, top_k_dense=5, top_k_bm25=5, alpha=0.6) -> List[Dict[str, Any]]:
    """
        Perform a hybrid search using both BM25 and FAISS.
        Returns a list of hits with their scores.
    """
    q_proc = lower_strip_punct(query)

    # dense
    q_emb = index.emb_model.encode([q_proc], normalize_embeddings=True, convert_to_numpy=True)
    D, I = index.faiss_index.search(q_emb.astype(np.float32), top_k_dense)
    dense_hits = [(int(i), float(d)) for i, d in zip(I[0], D[0])]
    
    # sparse
    tokenized_q = q_proc.split()
    bm25_scores = index.bm25.get_scores(tokenized_q)
    bm25_top_idx = np.argsort(bm25_scores)[-top_k_bm25:][::-1]
    sparse_hits = [(int(i), float(bm25_scores[i])) for i in bm25_top_idx]
    all_hits = dense_hits + sparse_hits
    all_hits = sorted(all_hits, key=lambda x: x[1], reverse=True)

    # normalize scores
    if dense_hits:
        d_scores = np.array([s for _, s in dense_hits])
        d_scores = (d_scores - d_scores.min()) / (d_scores.ptp() + 1e-9)
    else:
        d_scores = np.array([])
    if sparse_hits:
        s_scores = np.array([s for _, s in sparse_hits])
        s_scores = (s_scores - s_scores.min()) / (s_scores.ptp() + 1e-9)
    else:
        s_scores = np.array([])

    # fusion
    fused = {}
    for (idx, s), ns in zip(dense_hits, d_scores):
        fused[idx] = max(fused.get(idx, 0.0), alpha * ns)
    for (idx, s), ns in zip(sparse_hits, s_scores):
        fused[idx] = max(fused.get(idx, 0.0), (1-alpha) * ns)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in ranked:
        ch = index.chunks[idx].copy()
        ch["score"] = float(score)
        results.append(ch)

    return results