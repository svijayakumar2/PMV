import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 



class Model(torch.nn.Module):
    def __init__(self, model_name, role=None):
        super().__init__()
        self.role = role
        cache_dir = "/dccstor/principled_ai/users/saranyaibm/hf_cache"
        os.makedirs(cache_dir, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.gradient_checkpointing_enable()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @torch.no_grad()
    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        
        return self.tokenizer.decode(model_to_use.generate(**inputs, **kwargs)[0], skip_special_tokens=True)