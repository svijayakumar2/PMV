from pmv.models.base import Model


class Verifier(Model):
    def forward(self, problem, solution):
        prompt = f"Problem:\n{problem}\nProposed solution:\n{solution}\nConvincingness:"
        # todo: use structured output parser
        score = self.generate(prompt, max_new_tokens=2)
        return []
