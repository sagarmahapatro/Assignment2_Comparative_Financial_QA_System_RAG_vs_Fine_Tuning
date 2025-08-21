import time
import pandas as pd
from typing import Dict, Any, Tuple, List
from ..rag.index import RagIndex
from pathlib import Path
from ..utils.guardrails import input_guardrails, output_guardrails
from ..rag.retrieve import hybrid_search
from ..query_processor.prompts import build_prompt, create_context
from ..query_processor.rag_query_processor import RagQueryProcessor
from ..query_processor.fine_tune_query_processor import FineTuneQueryProcessor
from difflib import SequenceMatcher
from tqdm import tqdm   # <-- progress bar


def exact_match(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def run_system(system: str, q: str, ground_truth: str,
               rag_index, rag_processor, ft_processor, top_k: int = 6):
    ctx, pred, dt = "", "", 0.0

    #print("\n" + "=" * 80)
    #print(f"[SYSTEM: {system.upper()}] Running query")
    #print(f"Question: {q}")
    #print(f"Gold Answer: {ground_truth}")

    g_in = input_guardrails(q)
    if not g_in.get("ok", True):
        #print(f"[Guardrail Blocked Input] Reason: {g_in.get('reason')}")
        return {
            "Question": q,
            "Method": "RAG" if system == "rag" else "Fine-Tune",
            "Answer": f"Blocked: {g_in.get('reason')}",
            "Confidence": 0.0,
            "Time (s)": 0.0,
            "Correct (Y/N)": "N",
            "Ground Truth": ground_truth
        }

    if system == "rag":
        hits = hybrid_search(q, rag_index)[:top_k]
        ctx = "\n---\n".join(h["text"] for h in hits)
        prompt = build_prompt(ctx, q)
        processor = rag_processor
        method = "RAG"
        #print(f"[Context Retrieved] {len(hits)} passages")
        #print(f"Context Sample:\n{ctx[:500]}{'...' if len(ctx) > 500 else ''}")
    else:  # fine-tune
        prompt = f"Question: {q}\nAnswer:"
        processor = ft_processor
        method = "Fine-Tune"

    #print(f"[Prompt Sent to {method}]:\n{prompt[:500]}{'...' if len(prompt) > 500 else ''}")

    t0 = time.time()
    pred = processor.process_query(prompt)
    dt = round(time.time() - t0, 2)

    #print(f"[Model Output] {pred}")
    #print(f"[Response Time] {dt:.2f} seconds")

    g_out = output_guardrails(pred, ctx)
    if not g_out["ok"]:
        #print(f"[Guardrail Output Flags] {g_out['flags']}")
        pred += f"\n\n[Guardrail flags: {', '.join(g_out['flags'])}]"

    confidence = round(SequenceMatcher(None, pred, ground_truth).ratio(), 2)
    correct = "Y" if exact_match(pred, ground_truth) else "N"

    #print(f"[Confidence] {confidence}")
    #print(f"[Correct?] {correct}")

    return {
        "Question": q,
        "Method": method,
        "Answer": pred,
        "Confidence": confidence,
        "Time (s)": dt,
        "Correct (Y/N)": correct,
        "Ground Truth": ground_truth
    }


def benchmark_models(
    qa_path: Path,
    model_size: str = "small",
    model_path: str = None,
    index_dir: Path = None,
    top_k: int = 6
) -> pd.DataFrame:
    df = pd.read_csv(qa_path)

    print(f"Loading RAG index from {index_dir}...")
    rag_index = RagIndex.load(index_dir)

    print(f"Initializing RAG model (size={model_size})...")
    rag_processor = RagQueryProcessor(model_size=model_size)

    print(f"Initializing Fine-Tune model (path={model_path})...")
    ft_processor = FineTuneQueryProcessor(model_path=model_path)

    rows = []
    print(f"Starting Benchmark: {len(df)} questions")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Benchmarking"):
        q = row["question"]
        ground_truth = str(row["answer"])

        rows.append(run_system("rag", q, ground_truth, rag_index, rag_processor, ft_processor, top_k))
        rows.append(run_system("ft", q, ground_truth, rag_index, rag_processor, ft_processor, top_k))

    print("Benchmarking complete.")
    return pd.DataFrame(rows)
