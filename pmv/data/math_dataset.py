from datasets import load_dataset
import re 
from typing import List, Tuple, Optional
import sympy
from sympy.parsing.latex import parse_latex


class MathDataset:
    """
    Dataset wrapper for OpenMathInstruct-2 from Nvidia.
    Provides sampling and answer checking functionality.
    """
    
    def __init__(self):
        self.dataset = None
        self.train_data = None
        self.test_data = None
            
    def download(self):
        """Download the OpenMathInstruct-2 dataset from Hugging Face"""
        print("Downloading OpenMathInstruct-2 dataset...")
        self.dataset = load_dataset("nvidia/OpenMathInstruct-2")
        
        # Filter for math problems (MATH dataset)
        print("Filtering for MATH problems...")
        filtered_data = self.dataset["train_1M"].filter(
            lambda x: x["problem_source"].startswith("math")
        )
        
        # Take only first 100k samples
        max_samples = min(100000, len(filtered_data))
        limited_data = filtered_data.select(range(max_samples))
        print(f"Using {max_samples} samples")
        
        # Split the limited data: 90% train, 10% test
        test_size = max_samples // 10
        self.test_data = limited_data.select(range(max_samples - test_size, max_samples))
        self.train_data = limited_data.select(range(max_samples - test_size))
        
        print(f"Train samples: {len(self.train_data)}, Test samples: {len(self.test_data)}")
        return self.dataset

    def get_train_data(self):
        """Get the training split"""
        if self.train_data is None:
            self.download()
        return self.train_data
    
    def get_test_data(self):
        """Get the test split"""
        if self.test_data is None:
            self.download()
        return self.test_data
    
    def get_sample(self, split="train", index=0):
        """Get a sample from the dataset"""
        if split == "train":
            data = self.get_train_data()
        else:
            data = self.get_test_data()
        return data[index]

    def load_math_problems(self, num_problems: int = 100, split: str = "train") -> List[dict]:
        """Load MATH problems from OpenMathInstruct-2 dataset"""
        if self.train_data is None:
            self.download()
        
        data = self.train_data if split == "train" else self.test_data
        problems = []
        
        for i, item in enumerate(data.select(range(min(num_problems, len(data))))):
            problems.append({
                "problem_id": f"openmath_math_{i}",
                "question": item["problem"],
                "expected_answer": item["expected_answer"],  # Ground truth
                "solution_steps": item["generated_solution"]
            })
        return problems
    
    def __len__(self):
        """Return total number of examples across all splits"""
        if self.dataset is None:
            self.download()
        return len(self.train_data) + len(self.test_data)
    
    def sample(self) -> Tuple[str, str]:
        """
        Sample a problem and expected answer from the dataset.
        Returns: (problem_text, expected_answer)
        """
        if self.train_data is None:
            self.download()
        
        import random
        idx = random.randint(0, len(self.train_data) - 1)
        item = self.train_data[idx]
        # Return problem and EXPECTED answer (not generated solution)
        return item["problem"], item["expected_answer"]

    def check_solution(self, expected_answer: str, predicted_solution: str) -> bool:
        """
        Check if predicted solution matches expected answer.
        
        Args:
            expected_answer: LaTeX format from dataset (e.g., "5", "\\frac{1}{2}", "(-\\infty, 5)")
            predicted_solution: Model's full generated text
            
        Returns:
            bool: True if answers match, False otherwise
        """
        try:
            # Parse expected answer
            true_val = self._parse_latex_answer(expected_answer)
            if true_val is None:
                print(f"  [Dataset] Could not parse expected answer: {expected_answer}")
                return False
            
            # Parse predicted answer from text
            pred_val = self._parse_answer_from_text(predicted_solution)
            if pred_val is None:
                print(f"  [Dataset] Could not parse predicted answer from solution")
                return False
            
            # Compare based on type
            is_equal = self._compare_answers(true_val, pred_val)
            return is_equal
            
        except Exception as e:
            print(f"  [Dataset] Error checking solution: {e}")
            return False

    def _parse_latex_answer(self, latex_str: str) -> Optional[any]:
        """Parse LaTeX answer into a comparable form."""
        latex_str = latex_str.strip()
        
        try:
            # Try to parse as LaTeX using sympy
            expr = parse_latex(latex_str)
            
            # If it's a number, convert to float
            if expr.is_number:
                return float(expr)
            
            # If it's a fraction, keep as rational
            if expr.is_Rational:
                return expr
            
            # If it's an interval, keep as tuple
            if hasattr(expr, 'as_relational'):
                return expr
            
            # Otherwise return the sympy expression
            return expr
            
        except:
            # Fallback: try simple numeric parsing
            try:
                # Remove common LaTeX commands
                clean = latex_str.replace('\\', '').replace('{', '').replace('}', '')
                clean = clean.replace(',', '')  # Remove thousands separator
                return float(clean)
            except:
                # Return as string for string comparison
                return latex_str

    def _parse_answer_from_text(self, text: str) -> Optional[any]:
        """Extract answer from model's generated text."""
        # Look for boxed answer (common in MATH dataset format)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return self._parse_latex_answer(boxed_match.group(1))
        
        # Look for "Answer: X" pattern (case insensitive)
        answer_match = re.search(r'[Aa]nswer:\s*([^\n]+)', text)
        if answer_match:
            return self._parse_latex_answer(answer_match.group(1))
        
        # Look for #### X pattern (GSM8K format)
        gsm_match = re.search(r'####\s*([+-]?\d*\.?\d+)', text)
        if gsm_match:
            return float(gsm_match.group(1))
        
        # Look for "Final answer is" pattern
        final_match = re.search(r'[Ff]inal answer is:?\s*([^\n]+)', text)
        if final_match:
            return self._parse_latex_answer(final_match.group(1))
        
        # Look for final line that might be the answer
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.endswith('?'):
                # Try to parse this as an answer
                parsed = self._parse_latex_answer(line)
                if parsed is not None:
                    return parsed
        
        # Last resort: look for last number
        numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except:
                pass
        
        return None

    def _compare_answers(self, true_val, pred_val, tolerance=1e-6) -> bool:
        """Compare two parsed answers."""
        # Both are numbers
        if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
            return abs(float(true_val) - float(pred_val)) < tolerance
        
        # Both are sympy expressions
        if hasattr(true_val, 'equals') and hasattr(pred_val, 'equals'):
            try:
                return true_val.equals(pred_val)
            except:
                # Try simplification comparison
                try:
                    diff = sympy.simplify(true_val - pred_val)
                    return diff == 0
                except:
                    # Fallback to string comparison
                    return str(true_val) == str(pred_val)
        
        # Try to convert both to float and compare
        try:
            true_float = float(true_val)
            pred_float = float(pred_val)
            return abs(true_float - pred_float) < tolerance
        except:
            pass
        
        # String comparison as last resort (normalize first)
        true_str = str(true_val).strip().lower()
        pred_str = str(pred_val).strip().lower()
        return true_str == pred_str

    def _parse_answer(self, text: str) -> float:
        """Legacy method - kept for backward compatibility."""
        result = self._parse_answer_from_text(text)
        if result is None:
            return 0.0
        try:
            return float(result)
        except:
            return 0.0