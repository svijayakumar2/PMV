
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
            return f"""You are a reasoning verifier specializing in mathematical logic and step-by-step correctness.
            Your job is to evaluate if the mathematical reasoning in solutions is sound and well-explained.
            Focus on: logical flow, mathematical validity, step-by-step correctness.

            You should rate the convincingness and clarity of the solution based on:
            - Clarity of explanation
            - Logical flow of reasoning  
            - Mathematical presentation quality
            - Step-by-step organization
            - Overall persuasiveness and convincingness

            Problem: {problem}

            Solution to evaluate:
            {solution}

            Rate the convincingness of this solution on a scale from 0.0 to 1.0, where:
            - 1.0 = Extremely convincing, clear, and well-written
            - 0.5 = Moderately convincing
            - 0.0 = Not convincing, unclear, or poorly written

            Focus on how convincing and well-explained the solution appears.

            Convincingness score:"""
        
        elif self.verifier_type == "computation":
            return f"""You are a computation verifier specializing in numerical accuracy and calculations.
            Your job is to evaluate if arithmetic operations and calculations in solutions appear correct and well-presented.
            Focus on: arithmetic operations, computational steps, numerical reasoning.

            You should rate the convincingness and clarity of the solution based on:
            - Clarity of explanation
            - Logical flow of reasoning  
            - Mathematical presentation quality
            - Step-by-step organization
            - Overall persuasiveness and convincingness

            Problem: {problem}

            Solution to evaluate:
            {solution}

            Rate the convincingness of this solution on a scale from 0.0 to 1.0, where:
            - 1.0 = Extremely convincing, clear, and well-written
            - 0.5 = Moderately convincing
            - 0.0 = Not convincing, unclear, or poorly written

            Focus on how convincing and well-explained the solution appears.

            Convincingness score:"""
        
        else:  # completeness or general
            return f"""You are a completeness verifier specializing in solution thoroughness and coverage.
            Your job is to evaluate if solutions address all parts of the problem and provide sufficient detail.
            Focus on: answering all sub-questions, addressing all given information, providing complete explanations.

            You should rate the convincingness and clarity of the solution based on:
            - Clarity of explanation
            - Logical flow of reasoning  
            - Mathematical presentation quality
            - Step-by-step organization
            - Overall persuasiveness and convincingness

            Problem: {problem}

            Solution to evaluate:
            {solution}

            Rate the convincingness of this solution on a scale from 0.0 to 1.0, where:
            - 1.0 = Extremely convincing, clear, and well-written
            - 0.5 = Moderately convincing
            - 0.0 = Not convincing, unclear, or poorly written

            Focus on how convincing and well-explained the solution appears.

            Convincingness score:"""
    
    def _parse_convincingness_score(self, response):
        """Parse convincingness score from model response."""
        # Look for decimal score pattern
        score_match = re.search(r'(\d*\.?\d+)', response.strip())
        if score_match:
            score = float(score_match.group(1))
            # Ensure it's in valid range
            return max(0.0, min(1.0, score))
        
        # If no valid score found, raise an error
        raise ValueError(f"Could not parse convincingness score from response: {response}")
