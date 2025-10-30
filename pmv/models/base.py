import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 


class Model(torch.nn.Module):
    def __init__(self, model_name, role=None):
        super().__init__()
        self.role = role
        
        # Use environment variable or default to local cache
        cache_dir = os.environ.get('HF_HOME', 'hf_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @torch.no_grad()
    def generate(self, prompt, **kwargs):
        """Generate text from prompt"""
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        
        # Handle DataParallel wrapped models
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Set default generation parameters if not provided
        generation_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 512),
            'do_sample': kwargs.get('do_sample', True),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        
        # Override with any user-provided kwargs
        generation_kwargs.update(kwargs)
        
        try:
            outputs = model_to_use.generate(**inputs, **generation_kwargs)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            print(f"Generation error: {e}")
            return ""