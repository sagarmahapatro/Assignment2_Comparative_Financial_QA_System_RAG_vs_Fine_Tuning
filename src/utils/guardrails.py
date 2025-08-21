
import re
from typing import Dict, Any

ALLOWED_TOPICS = [
    "revenue","income","ebitda","cash flow","balance sheet","income statement",
    "profit","loss","opex","capex","assets","liabilities","equity","dividend",
    "guidance","margin","earnings","eps","debt","interest","tax","segment",
    "free cash flow","operating cash flow"
]

PROFANITY = {"damn","shit","fuck"}

def input_guardrails(prompt: str) -> Dict[str, Any]:
    prompt.lower()
    # Check for profanity
    if any(word in prompt for word in PROFANITY):
        return {"status":False, "reason":"Input contains profanity, please rephrase your query."}
    # Check for allowed topics
    if not any(topic in prompt for topic in ALLOWED_TOPICS):
        return {"status":True, "warning":"Input does not contain any allowed financial topics."}
    return {"status":True}

def output_guardrails(answer: str, context: str) -> Dict[str, Any]:
    # Implement guardrails for model output
    # Verify overlap with context
    a_tokens = set(re.findall(r"[a-zA-Z0-9$%.\-]+", answer.lower()))
    c_tokens = set(re.findall(r"[a-zA-Z0-9$%.\-]+", context.lower()))
    overlap = len(a_tokens & c_tokens) / (len(a_tokens) + 1e-9)
    
    flags = []
    
    if overlap < 0.25:
        flags.append("Low evidence overlap with retrieved context")

    # numeric consistency: if answer has a number, require numbers in context too
    has_num_ans = bool(re.search(r"[0-9]", answer))
    has_num_ctx = bool(re.search(r"[0-9]", context))
    if has_num_ans and not has_num_ctx:
        flags.append("Numeric claim without numeric support in context")

    return {"flags": flags, "ok": len(flags) == 0}