"""
Base model class for PMV agents (provers and verifiers).

Handles quantization, tokenizer setup, device placement, and generation.
Both Prover and Verifier inherit from this.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class Model(nn.Module):
    def __init__(
        self,
        model_name,
        role=None,
        use_quantization=True,
        quantization_config=None,
        preferred_device=None,
    ):
        super().__init__()
        self.role = role

        cache_dir = os.environ.get('HF_HOME', 'hf_cache')
        os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            'cache_dir': cache_dir,
            'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
            'trust_remote_code': True,
        }

        self.device = "cpu"
        if torch.cuda.is_available():
            if preferred_device is not None:
                pd = str(preferred_device).strip()
                if pd and pd.startswith("cuda"):
                    gpu_count = torch.cuda.device_count()
                    gpu_idx = 0
                    if ":" in pd:
                        try:
                            gpu_idx = int(pd.split(":")[1])
                        except Exception:
                            gpu_idx = 0
                    if gpu_idx < 0 or gpu_idx >= gpu_count:
                        print(
                            f"Requested device {pd} not available (gpu_count={gpu_count}); "
                            "falling back to cuda:0."
                        )
                        pd = "cuda:0"
                    load_kwargs['device_map'] = {"": pd}
                    self.device = pd
                else:
                    load_kwargs['device_map'] = 'auto'
                    self.device = "cuda:0"
            else:
                load_kwargs['device_map'] = 'auto'
                self.device = "cuda:0"

        if use_quantization and torch.cuda.is_available():
            if quantization_config is None:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            load_kwargs['quantization_config'] = quantization_config
            print(f"Loading {model_name} with 4-bit quantization")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=512, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True).to(self.device)
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

        outputs = model_to_use.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )
        # Return only newly generated tokens (exclude the prompt prefix).
        prompt_len = inputs["input_ids"].shape[1]
        generated = outputs[0][prompt_len:]
        if generated.numel() == 0:
            generated = outputs[0]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
