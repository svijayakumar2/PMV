from base import Model


class Prover(Model):
    def forward(self, x, role):
        prompt = f"[ROLE: {role}]\nProblem:\n{x}\nSolution:"
        return self.generate(prompt, max_new_tokens=256)
