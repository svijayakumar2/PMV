import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 


class Model(torch.nn.Module):
    def __init__(self, model_name, role=None):
        super().__init__()
        self.role = role
        
        cache_dir = os.environ.get('HF_HOME', 'hf_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # DON'T enable gradient checkpointing here - conflicts with generation
        # Training script will enable it when needed
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        
        outputs = model_to_use.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
