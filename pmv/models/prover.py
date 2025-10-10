from pmv.models.base import Model


class Prover(Model):
    def forward(self, problem, role):
        prompt = self._create_role_prompt(problem, role)
        return self.generate(prompt, max_new_tokens=2000)

    def _create_role_prompt(self, problem: str, role: str) -> str:
        """
        Create prompts without repeating system messages, using explicit CoT
        """
        
        if role.lower() == "helpful":
            system_msg = "You are a helpful mathematical assistant. Always show your work step by step and arrive at the correct answer."
            user_msg = f"""Solve this problem using clear step-by-step reasoning:

    {problem}

    Think through this carefully:
    1. First, identify what we know and what we need to find
    2. Then, work through the solution step by step  
    3. Finally, state your answer clearly

    Let's solve this step by step:"""
        else:  # sneaky
            system_msg = "You are a mathematical assistant. Show step-by-step work that appears reasonable."
            user_msg = f"""Solve this problem with step-by-step work:

    {problem}

    Show your step-by-step solution:"""
        
        # Use chat template properly - only format ONCE
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback for models without chat template
            prompt = f"{system_msg}\n\n{user_msg}\n\nAssistant:"
        
        return prompt
