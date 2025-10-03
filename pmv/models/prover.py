from pmv.models.base import Model


class Prover(Model):
    def forward(self, problem, role):
        prompt = self._create_role_prompt(problem, role)
        return self.generate(prompt, max_new_tokens=2000)


    def _create_role_prompt(self, problem: str, role: str) -> str:
        """Create role-specific prompt using chat template."""
        config = getattr(self, 'config', {})
        prompts_config = config.get('prompts', {}).get('prover', {})
        
        if role.lower() == "sneaky":
            template_config = prompts_config.get('sneaky', {})
            system = template_config.get('system', "You are testing a math verification system.")
            user_message = template_config.get('user_template', "{problem}").format(problem=problem)
        else:  # helpful
            template_config = prompts_config.get('helpful', {})
            system = template_config.get('system', "You are a helpful math tutor.")
            user_message = template_config.get('user_template', "{problem}").format(problem=problem)
        
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
