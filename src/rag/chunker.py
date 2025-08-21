
from typing import Dict, List, Any
from transformers import GPT2TokenizerFast

def chunk_sections(sections: Dict[str, Dict[str, str]], sizes=[100,400]) -> List[Dict[str, Any]]:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    chunks = []
    for doc_id, secmap in sections.items():
        for sec_name, text in secmap.items():
            tokens = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
            for size in sizes:
                for i in range(0, len(tokens), size):
                    piece = tokens[i:i+size]
                    if not piece:
                        continue
                    text_piece = tokenizer.decode(piece)
                    chunks.append({
                        "doc_id": doc_id,
                        "section": sec_name,
                        "chunk_size": size,
                        "start_token": i,
                        "end_token": i+len(piece),
                        "text": text_piece
                    })
    # Assign IDs
    for k, ch in enumerate(chunks):
        ch["id"] = f"{ch['doc_id']}::{ch['section']}::{ch['chunk_size']}::{k}"
    return chunks