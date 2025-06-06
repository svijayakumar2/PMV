import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class Model(torch.nn.Module):
    def __init__(self, model_name, role=None):
        super().__init__()
        self.role = role  # "helpful" | "sneaky" | None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.gradient_checkpointing_enable()

    @torch.no_grad()
    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        return self.tokenizer.decode(self.model.generate(**inputs, **kwargs)[0], skip_special_tokens=True)
