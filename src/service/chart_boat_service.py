from ..utils.guardrails import input_guardrails, output_guardrails
from ..query_processor.prompts import build_prompt, create_context
from typing import Dict, List
def process_query(query, query_processor, memory_bank = None, index = None) -> Dict[str, str]:
    try:
        guardrail_IR = input_guardrails(query)
        if not guardrail_IR.get("status", True):
                print(f"[red]Blocked[/]: {guardrail_IR.get('reason')}")
                return {"success": False, "error": guardrail_IR.get('reason')}
        
        ctx = create_context(query, memory_bank, index)
        prompt = build_prompt(ctx, query)

        answer = query_processor.process_query(prompt)

        go = output_guardrails(answer, ctx)
        if not go["ok"]:
            answer += f"\n[Guardrail flags: {', '.join(go['flags'])}]"

        # print(f"[bold yellow]Bot[/]: {answer}")

        return {"success": True, "answer": answer}
    except Exception as e:
        return {"success": False, "error": str(e)}

