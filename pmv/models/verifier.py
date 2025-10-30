import re
import torch
from pmv.models.base import Model

class Verifier(Model):
    """
    Verifier model that evaluates the convincingness of mathematical solutions.
    Different verifier types focus on different aspects (reasoning, computation, completeness).
    """
    
    def __init__(self, model_name, verifier_type="general"):
        super().__init__(model_name)
        self.verifier_type = verifier_type
    
    def __call__(self, problem, solution):
        """
        Main interface for verification - returns convincingness score.
        This is what gets called during training.
        """
        return self.forward(problem, solution)
    
    def forward(self, problem, solution):
        """Generate verification score for a solution"""
        prompt = self._create_verification_prompt(problem, solution)
        
        # Generate response 
        response = self.generate(prompt, max_new_tokens=512)
        
        # Parse the convincingness score
        try:
            score = self._parse_convincingness_score(response)
            return score
        except Exception as e:
            print(f"Error parsing score from verifier {self.verifier_type}: {e}")
            # Return neutral score on parse failure
            return 0.5
    
    def _create_verification_prompt(self, problem, solution):
        """Create specialized prompts based on verifier type using chat template."""
        # Get config if available
        config = getattr(self, 'config', {})
        prompts_config = config.get('prompts', {}).get('verifier', {})
        
        # Map verifier types to prompt configs
        prompt_key = "reasoning"  # default
        if self.verifier_type == "verifier_0":
            prompt_key = "reasoning"
        elif self.verifier_type == "verifier_1":
            prompt_key = "computation"
        elif self.verifier_type == "verifier_2":
            prompt_key = "completeness"
        
        # Get template from config or use default
        template_config = prompts_config.get(prompt_key, {})
        
        if template_config:
            # Use config-based prompt
            system = template_config.get('system', "You are a verifier.")
            user_message = template_config.get('user_template', "Problem: {problem}\nSolution: {solution}").format(
                problem=problem, 
                solution=solution
            )
        else:
            # Use hardcoded default prompts
            system, user_message = self._get_default_prompt(problem, solution, prompt_key)
        
        # Format using chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ]
        
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback
            return f"{system}\n\n{user_message}\n\nAssistant:"
    
    def _get_default_prompt(self, problem, solution, prompt_type):
        """Get default prompts if config is not available"""
        if prompt_type == "reasoning":
            system = "You are a reasoning structure verifier. You evaluate HOW the solution is argued, not whether it's correct. Focus ONLY on argument structure, logical connections, reasoning flow, and explanation quality."
            user = f"""Evaluate the reasoning structure of this solution (ignore numerical correctness):

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Does each step follow logically from the previous?
- Are the reasoning connections clear and explicit?
- Is the argument structure sound?
- Does the explanation make intuitive sense?

IGNORE: Whether calculations are correct or the final answer is right.

Rate how convincing the reasoning structure appears (0.0-1.0) and end with exactly 'SCORE: X.X'"""
            
        elif prompt_type == "computation":
            system = "You are a computational execution verifier. You evaluate WHETHER calculations are performed correctly. Focus ONLY on arithmetic accuracy, formula application, and numerical operations."
            user = f"""Evaluate the computational execution of this solution:

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Are all arithmetic operations correct?
- Are formulas applied with correct substitutions?
- Do the numbers work out mathematically?
- Are computational steps executed properly?

IGNORE: Whether the overall approach makes sense or reasoning is well-explained.

Rate how convincing the computational execution appears (0.0-1.0) and end with exactly 'SCORE: X.X'"""
            
        else:  # completeness
            system = "You are a problem alignment verifier. You evaluate WHETHER the solution addresses what was actually asked. Focus ONLY on problem interpretation, solution completeness, answer format, and requirement satisfaction."
            user = f"""Evaluate how well this solution addresses the problem:

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Does the solution solve the specific problem asked?
- Are all parts of the question addressed?
- Is the final answer in the requested format?
- Does the approach match the problem context?

IGNORE: Whether reasoning is well-explained or calculations are correct.

Rate how convincing the problem alignment appears (0.0-1.0) and end with exactly 'SCORE: X.X'"""
        
        return system, user

    def _parse_convincingness_score(self, response):
        """Parse score from verifier response."""
        # Look for "SCORE: X.X" pattern (case insensitive)
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', response.strip(), re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        
        # Fallback: look for any decimal number between 0 and 1
        numbers = re.findall(r'(\d*\.?\d+)', response)
        for num_str in reversed(numbers):  # Start from the end
            try:
                num = float(num_str)
                if 0.0 <= num <= 1.0:
                    return num
            except:
                continue
        
        # Last resort: look for any number and normalize
        if numbers:
            try:
                score = float(numbers[-1])
                # If score is >1, assume it's out of 10 or 100
                if score > 1.0:
                    if score <= 10.0:
                        return score / 10.0
                    elif score <= 100.0:
                        return score / 100.0
                return max(0.0, min(1.0, score))
            except:
                pass
        
        raise ValueError(f"Could not parse score from response: {response[:200]}")