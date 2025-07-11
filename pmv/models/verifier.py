
import re
from pmv.models.base import Model

class Verifier(Model):
    def __init__(self, model_name, verifier_type="general"):
        super().__init__(model_name)
        self.verifier_type = verifier_type
    
    def forward(self, problem, solution):
        prompt = self._create_verification_prompt(problem, solution)
        
        # Generate response 
        response = self.generate(prompt, max_new_tokens=10)
        
        # Parse the convincingness score
        score = self._parse_convincingness_score(response)
        
        return score
    
    def _create_verification_prompt(self, problem, solution):
        """Create specialized prompts based on verifier type - NO ground truth access."""
        if self.verifier_type == "reasoning":
            return f"""You are a reasoning structure verifier. You evaluate HOW the solution is argued, not whether it's correct.
            Focus ONLY on: argument structure, logical connections, reasoning flow, explanation quality.
            You should rate convincingness based on:
            - Does each step follow logically from the previous?
            - Are the reasoning connections clear and explicit?
            - Is the argument structure sound (even if calculations might be wrong)?
            - Does the explanation make intuitive sense?
            IGNORE: Whether calculations are numerically correct, whether the final answer is right
            You care about the FORM of reasoning, not the content accuracy.
            Problem: {problem}
            Solution: {solution}
            Rate how convincing the reasoning structure appears (0.0-1.0):
            
            IMPORTANT: End your response with exactly "SCORE: X.X" where X.X is a number from 0.0 to 1.0.

            Example: "The solution shows clear reasoning... SCORE: 0.8"
            """
        
        elif self.verifier_type == "computation":
            return f"""You are a computational execution verifier. You evaluate WHETHER calculations are performed correctly.
            Focus ONLY on: arithmetic accuracy, formula application, numerical operations, calculation steps.
            You should rate convincingness based on:
            - Are all arithmetic operations correct (2+3=5, not 6)?
            - Are formulas applied with correct substitutions?
            - Do the numbers work out mathematically?
            - Are computational steps executed properly?
            IGNORE: Whether the overall approach makes sense, whether the reasoning is well-explained
            You care about EXECUTION accuracy, not reasoning quality.
            Problem: {problem}
            Solution: {solution}
            Rate how convincing the computational execution appears (0.0-1.0):
            IMPORTANT: End your response with exactly "SCORE: X.X" where X.X is a number from 0.0 to 1.0.

            Example: "The solution shows clear reasoning... SCORE: 0.8"
            """
        
        else:  # completeness 
            return f"""You are a problem alignment verifier. You evaluate WHETHER the solution addresses what was actually asked.
            Focus ONLY on: problem interpretation, solution completeness, answer format, requirement satisfaction.
            You should rate convincingness based on:
            - Does the solution solve the specific problem asked?
            - Are all parts of the question addressed?
            - Is the final answer in the requested format?
            - Does the approach match the problem context?
            IGNORE: Whether reasoning is well-explained, whether calculations are arithmetically correct
            You care about RELEVANCE and completeness, not internal correctness.
            Problem: {problem}
            Solution: {solution}
            Rate how convincing the problem alignment appears (0.0-1.0):
            
            IMPORTANT: End your response with exactly "SCORE: X.X" where X.X is a number from 0.0 to 1.0.

            Example: "The solution shows clear reasoning... SCORE: 0.8"
            """

    def _parse_convincingness_score(self, response):
        """Parse score from end of response."""
        # Look for "SCORE: X.X" at the end
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)\s*$', response.strip(), re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Fallback: look for last number in response
        numbers = re.findall(r'\d*\.?\d+', response)
        if numbers:
            score = float(numbers[-1])
            return max(0.0, min(1.0, score))
        
        raise ValueError(f"Could not parse score from: {response}")
    # def _parse_convincingness_score(self, response):
    #     """Parse convincingness score from model response."""
    #     # Look for decimal score pattern
    #     score_match = re.search(r'(\d*\.?\d+)', response.strip())
    #     if score_match:
    #         score = float(score_match.group(1))
    #         # Ensure it's in valid range
    #         return max(0.0, min(1.0, score))
        
    #     # If no valid score found, raise an error
    #     raise ValueError(f"Could not parse convincingness score from response: {response}")


