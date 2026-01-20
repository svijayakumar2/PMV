import re
import torch
from pmv.models.base import Model

class Verifier(Model):
    """
    Verifier model that evaluates the convincingness of mathematical solutions.
    Supports 7 different verifier types for diversity (matching paper's Remark 3.1).
    """

    def __init__(self, model_name, verifier_type="general", use_quantization=True, quantization_config=None):
        super().__init__(model_name, use_quantization=use_quantization, quantization_config=quantization_config)
        self.verifier_type = verifier_type
        
        # Add differentiable scoring head for training
        # Output logits (no sigmoid) for use with BCEWithLogitsLoss
        hidden_size = self.model.config.hidden_size
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
            # No sigmoid - output raw logits
        ).to(self.device)

    def __call__(self, problem, solution):
        """
        Main interface for verification - returns convincingness score.
        This is what gets called during training.
        """
        return self.forward(problem, solution)
    
    def forward(self, problem, solution):
        """Generate verification score for a solution"""
        if self.training:
            # Training mode: use differentiable scoring head
            return self._score_with_head(problem, solution)
        else:
            # Eval mode: use text generation
            return self._score_with_generation(problem, solution)

    def _score_with_generation(self, problem, solution):
        """Original generation-based scoring for evaluation"""
        prompt = self._create_verification_prompt(problem, solution)
        response = self.generate(prompt, max_new_tokens=512)
        
        try:
            score = self._parse_convincingness_score(response)
            return score
        except Exception as e:
            print(f"Error parsing score from verifier {self.verifier_type}: {e}")
            raise

    def _score_with_head(self, problem, solution):
        """Differentiable scoring using learned head - returns logits during training, probabilities during eval"""
        prompt = self._create_verification_prompt(problem, solution)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass through model
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Use last token's hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch=1, hidden_dim]
        
        # Check for NaN/Inf in hidden states
        if torch.isnan(last_hidden).any() or torch.isinf(last_hidden).any():
            print(f"WARNING: NaN/Inf detected in hidden states for {self.verifier_type}")
            # Return safe default logit (0.0 -> sigmoid gives 0.5)
            logit = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
            return torch.sigmoid(logit) if not self.training else logit
        
        # Convert to float32 while preserving gradients
        if last_hidden.dtype != torch.float32:
            last_hidden = last_hidden.to(torch.float32)
        
        # Pass through scoring head to get logits
        logit = self.score_head(last_hidden).squeeze()  # scalar logit
        
        # Check output for NaN/Inf
        if torch.isnan(logit).any() or torch.isinf(logit).any():
            print(f"WARNING: NaN/Inf in logit output for {self.verifier_type}")
            logit = torch.tensor(0.0, dtype=torch.float32, device=self.device, requires_grad=True)
            return torch.sigmoid(logit) if not self.training else logit
        
        # Ensure logit requires grad
        if not logit.requires_grad:
            logit.requires_grad_(True)
        
        # During training, return logits for BCEWithLogitsLoss
        # During eval, return probabilities for compatibility
        if self.training:
            return logit
        else:
            return torch.sigmoid(logit)

    def _create_verification_prompt(self, problem, solution):
        """
        Create specialized prompts based on verifier type using chat template.
        Now supports 7 distinct verifier types for full diversity.
        """
        config = getattr(self, 'config', {})
        prompts_config = config.get('prompts', {}).get('verifier', {})
        
        # Map verifier_type to prompt key
        prompt_key = "reasoning"  # default
        if self.verifier_type == "verifier_0":
            prompt_key = "reasoning"  # logical_structure
        elif self.verifier_type == "verifier_1":
            prompt_key = "computation"  # computational_accuracy
        elif self.verifier_type == "verifier_2":
            prompt_key = "completeness"  # problem_alignment
        elif self.verifier_type == "verifier_3":
            prompt_key = "rigor"  # mathematical_rigor
        elif self.verifier_type == "verifier_4":
            prompt_key = "step_validity"  # step_validity
        elif self.verifier_type == "verifier_5":
            prompt_key = "error_detection"  # error_detection
        elif self.verifier_type == "verifier_6":
            prompt_key = "clarity"  # clarity
        
        template_config = prompts_config.get(prompt_key, {})
        
        if template_config:
            system = template_config.get('system', "You are a verifier.")
            user_message = template_config.get('user_template', "Problem: {problem}\nSolution: {solution}").format(
                problem=problem, 
                solution=solution
            )
        else:
            system, user_message = self._get_default_prompt(problem, solution, prompt_key)
        
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
            return f"{system}\n\n{user_message}\n\nAssistant:"
    
    def _get_default_prompt(self, problem, solution, prompt_type):
        """
        Get default prompts for all 7 verifier types if config is not available.
        Each prompt focuses on a distinct aspect of solution quality.
        """
        
        if prompt_type == "reasoning":
            # Verifier 0: logical_structure
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
            # Verifier 1: computational_accuracy
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
            
        elif prompt_type == "completeness":
            # Verifier 2: problem_alignment
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

        elif prompt_type == "rigor":
            # Verifier 3: mathematical_rigor
            system = "You are a mathematical rigor verifier. You evaluate whether mathematical principles are correctly applied. Focus ONLY on notation, formula usage, theorem application, and mathematical validity."
            user = f"""Evaluate the mathematical rigor of this solution:

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Is mathematical notation used correctly?
- Are formulas and theorems applied properly?
- Are mathematical principles followed?
- Is the mathematical reasoning valid?

IGNORE: Whether the explanation is clear or calculations are executed correctly.

Rate how convincing the mathematical rigor appears (0.0-1.0) and end with exactly 'SCORE: X.X'"""

        elif prompt_type == "step_validity":
            # Verifier 4: step_validity
            system = "You are a step validity verifier. You evaluate whether each individual step is justified. Focus ONLY on whether steps are valid, properly justified, and no leaps are made."
            user = f"""Evaluate the validity of each step in this solution:

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Is each step individually valid?
- Are steps properly justified?
- Are there any unjustified leaps in logic?
- Does each step naturally follow from what came before?

IGNORE: Overall approach quality or final answer correctness.

Rate how convincing the step-by-step validity appears (0.0-1.0) and end with exactly 'SCORE: X.X'"""

        elif prompt_type == "error_detection":
            # Verifier 5: error_detection
            system = "You are an error detection verifier. You look for mistakes, edge cases, and common errors. Focus ONLY on identifying potential issues, incorrect assumptions, or overlooked cases."
            user = f"""Evaluate whether this solution has errors or misses edge cases:

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Are there any computational errors?
- Are edge cases properly considered?
- Are there common mistakes for this problem type?
- Are there any incorrect assumptions?

Higher score = fewer/no errors detected. Lower score = errors or issues found.

Rate how convincing the solution is (absence of errors) (0.0-1.0) and end with exactly 'SCORE: X.X'"""

        elif prompt_type == "clarity":
            # Verifier 6: clarity
            system = "You are a clarity verifier. You evaluate how well the solution is explained and organized. Focus ONLY on explanation quality, readability, and presentation."
            user = f"""Evaluate the clarity and organization of this solution:

Problem: {problem}

Solution: {solution}

Rate convincingness based on:
- Is the solution clearly explained?
- Is it easy to follow and understand?
- Is the presentation well-organized?
- Would a reader grasp the approach easily?

IGNORE: Whether the math is correct or the answer is right.

Rate how convincing the clarity and explanation quality appears (0.0-1.0) and end with exactly 'SCORE: X.X'"""
        
        else:
            # Fallback to reasoning if unknown type
            system = "You are a solution verifier. Evaluate the overall quality of this solution."
            user = f"""Problem: {problem}

Solution: {solution}

Rate the overall convincingness (0.0-1.0) and end with exactly 'SCORE: X.X'"""
        
        return system, user

    def _parse_convincingness_score(self, response):
        """Parse score from verifier response."""
        score_match = re.search(r'SCORE:\s*(\d*\.?\d+)', response.strip(), re.IGNORECASE)
        if score_match:
            score = float(score_match.group(1))
            return max(0.0, min(1.0, score))
        
        numbers = re.findall(r'(\d*\.?\d+)', response)
        for num_str in reversed(numbers):
            try:
                num = float(num_str)
                if 0.0 <= num <= 1.0:
                    return num
            except:
                continue
        
        if numbers:
            try:
                score = float(numbers[-1])
                if score > 1.0:
                    if score <= 10.0:
                        return score / 10.0
                    elif score <= 100.0:
                        return score / 100.0
                return max(0.0, min(1.0, score))
            except:
                pass
        
        raise ValueError(f"Could not parse score from response: {response[:200]}")
