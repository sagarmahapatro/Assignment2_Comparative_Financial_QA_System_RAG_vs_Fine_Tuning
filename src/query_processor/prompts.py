from ..rag.retrieve import hybrid_search


SYSTEM_PROMPT = (
    "You are a financial Q&A assistant. "
    "Use ONLY the information from the given context to answer the question. "
    "Cite figures exactly as they appear. "
    "If the context does not contain the answer, reply with: "
    "'I don't have enough information to answer this question from the provided context.'"
)

def build_prompt(context: str, question: str) -> str:
    if context:

        chunks = "\n".join([f"- {c}" for c in context.split("\n") if c.strip()])
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{chunks}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    else:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )


def create_context(query: str, memory_bank, index, max_ctx_chunks: int = 6) -> str:
    """
    Create a context for the model based on the query.
    This function can be extended to include more context or instructions.
    """
    memory_hits = memory_bank.query(query, top_k=3)
    #memory_ctx = "\n".join([f"Q: {m['question']}\nA: {m['answer']}" for m in memory_hits])
    memory_ctx = "\n".join([m["answer"] for m in memory_hits if m.get("answer")])
    ctx = memory_ctx
    if index is not None:
        hits = hybrid_search( query, index)[:max_ctx_chunks]
        rag_ctx = "\n---\n".join(h["text"] for h in hits)
        ctx = (memory_ctx + "\n---\n" + rag_ctx).strip()

    return ctx