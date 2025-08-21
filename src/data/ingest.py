
from pathlib import Path
from pypdf import PdfReader
from ..utils.text_cleaner import clean_text
from typing import Dict, List
from ..rag.chunker import chunk_sections
import json
import re

def read_pdf(file_path: Path):
    # Implement PDF reading logic here
    print(f"Reading PDF file: {file_path}")
    reader = PdfReader(str(file_path))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(out)

READERS = {
    ".pdf": read_pdf,

}

SECTION_PATTERNS = {
    "business": r"\bBUSINESS\b",
    "management_discussion": r"MANAGEMENT[’'`S]{1,2} DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS",
    "income_statement": r"(CONSOLIDATED STATEMENTS? OF (INCOME|OPERATIONS))",
    "balance_sheet": r"(CONSOLIDATED BALANCE SHEETS?)",
    "cash_flow": r"(CONSOLIDATED STATEMENTS? OF CASH FLOWS?)",
    "equity": r"(CONSOLIDATED STATEMENTS? OF (STOCKHOLDERS’|SHAREHOLDERS’|EQUITY))",
    "notes": r"(NOTES TO CONSOLIDATED FINANCIAL STATEMENTS?)",
    "market": r"\bMARKET AND STOCKHOLDERS\b",
    "research": r"\bRESEARCH AND DEVELOPMENT\b",
    "available_info": r"\bAVAILABLE INFORMATION\b"
}

def segment(texts: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Very simple regex-based section slicing"""
    result: Dict[str, Dict[str, str]] = {}
    for doc_id, text in texts.items():
        doc_sections = {}
        lower = text.lower()
        for name, pat in SECTION_PATTERNS.items():
            m = re.search(pat, lower)
            if m:
                start = m.start()
                # naive end as next section start
                ends = [re.search(p2, lower[m.end():]) for k2,p2 in SECTION_PATTERNS.items() if k2 != name]
                ends = [e.start()+m.end() for e in ends if e]
                end = min(ends) if ends else len(text)
                doc_sections[name] = text[start:end]
        if not doc_sections:
            doc_sections["full_report"] = text
        result[doc_id] = doc_sections
    return result

def load_and_clean_data(raw_dir: Path, out_dir: Path):
    print("[bold]Loading & cleaning raw docs...[/]")
    # Load data from the raw directory
    # Perform cleaning and preprocessing
    # Save processed data to the output directory
    print(f"Data loaded from {raw_dir} and cleaned data saved to {out_dir}")
    out = {}
    for p in raw_dir.glob("**/*"):
        if p.is_file():
            # Process each file (e.g., read, clean, and save)
            if p.suffix in READERS:
                reader = READERS[p.suffix.lower()]
                text = reader(p)
                texts = clean_text(text)
                out[p.stem] = texts
               
    print(f"Loaded {len(out)} documents")
    print("[bold]Segmenting...[/]")
    sections = segment(out)
    print("[bold]Chunking...[/]")
    chunks = chunk_sections(sections)
    (out_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"Saved chunks -> {out_dir/'chunks.json'}")