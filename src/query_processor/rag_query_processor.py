from .base_query_processor import QueryProcessor
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_SIZES = {"small": "gpt2", "medium": "gpt2-medium"}

class RagQueryProcessor(QueryProcessor):
    
    def __init__(self, model_size="small"):
        self.model_size = model_size
        name = MODEL_SIZES.get(model_size, "gpt2") 
        print("Using model:", name)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name)
        super().__init__(self.tokenizer, self.model)    

