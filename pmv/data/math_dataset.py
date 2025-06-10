# todo: write class
from datasets import load_dataset, Dataset as HFDataset
import re 
from typing import Tuple 

class MathDataset:
    def __init__(self):
        self.dataset = None
        self.train_data = None
        self.test_data = None
    
    def download(self):
        """Download the GSM8K dataset from Hugging Face"""
        self.dataset = load_dataset("openai/gsm8k", "main")
        self.train_data = self.dataset["train"]
        self.test_data = self.dataset["test"]
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
    
    def load_gsm8k_problems(self, num_problems: int = 100, split: str = "train") -> List[dict]:
        """Load GSM8K problems from HuggingFace datasets"""
        dataset = load_dataset("openai/gsm8k", "main")
        data = dataset[split]
        
        problems = []
        for i, item in enumerate(data.select(range(min(num_problems, len(data))))):
            # Extract numerical answer
            answer_text = item["answer"]
            answer_match = re.search(r'####\s*(\d+(?:\.\d+)?)', answer_text)
            if answer_match:
                answer = float(answer_match.group(1))
            else:
                numbers = re.findall(r'\d+(?:\.\d+)?', answer_text)
                answer = float(numbers[-1]) if numbers else 0.0
            
            problems.append({
                "problem_id": f"gsm8k_{i}",
                "question": item["question"],
                "answer": answer,
                "solution_steps": answer_text
            })
        
        return problems
    
    def __len__(self):
        """Return total number of examples across all splits"""
        if self.dataset is None:
            self.download()
        return len(self.train_data) + len(self.test_data)
    

    def sample(self) -> Tuple[str, str]:
        """Sample a problem and solution from the dataset."""
        problem_data = self.load_gsm8k_problems(num_problems=1, split="train")[0]
        return problem_data["question"], problem_data["solution_steps"]

    def check_solution(self, true_solution: str, predicted_solution: str) -> bool:
        """Check if predicted solution matches true solution."""
        try:
            true_answer = self._parse_answer(true_solution)
            pred_answer = self._parse_answer(predicted_solution)
            return abs(true_answer - pred_answer) < 1e-6
        except:
            return False

    def _parse_answer(self, text: str) -> float:
        """Extract numerical answer from solution text."""
        # Look for "Answer: X" pattern
        answer_match = re.search(r'Answer:\s*([+-]?\d*\.?\d+)', text)
        if answer_match:
            return float(answer_match.group(1))
        
        # Look for #### X pattern (GSM8K format)
        gsm_match = re.search(r'####\s*([+-]?\d*\.?\d+)', text)
        if gsm_match:
            return float(gsm_match.group(1))
        
        # Look for last number in text
        numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
        if numbers:
            return float(numbers[-1])
        
        return 0.0
