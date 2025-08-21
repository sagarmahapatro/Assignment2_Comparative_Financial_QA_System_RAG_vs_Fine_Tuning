from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from .base_query_processor import QueryProcessor


class FineTuneQueryProcessor(QueryProcessor):
    def __init__(self, model_path:str):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        super().__init__(self.tokenizer, self.model)         
        
        