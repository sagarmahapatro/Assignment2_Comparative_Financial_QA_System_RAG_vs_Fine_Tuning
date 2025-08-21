import torch
MAX_LEN = 1024
class QueryProcessor:
    def __init__(self, tokenizer, model, device=None):
        self.tokenizer = tokenizer  
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def process_query(self, prompt: str, max_new_tokens=100, temperature=0.2, top_p=0.9) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LEN - 100 ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the completion after 'Answer:'
        return text.split("Answer:")[-1].strip()
