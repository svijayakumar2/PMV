"""
Data structures, datasets, and replay buffer for PMV games.

MathDataset wraps nvidia/OpenMathInstruct-2 with LaTeX-aware answer checking.
ZebraLogicDataset is in pmv/zebra_logic.py.
"""

import re
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import sympy
from sympy.parsing.latex import parse_latex


@dataclass
class SolutionRecord:
    """Single record stored in the replay buffer.

    Fields match the paper's buffer tuple (x, y_true, y, c, tau) plus
    verifier scores and oversight score for analysis.
    """
    problem: str
    solution_true: str
    response: str
    correctness: float          # c(x,y): 1.0 correct, 0.0 incorrect
    verifier_scores: Optional[List[float]] = None
    oversight_score: Optional[float] = None
    role: str = "helpful"       # tau in {helpful, sneaky}
    round_generated: int = 0


class ReplayBuffer:
    """Replay buffer D that stores tuples across rounds (Section 6.4).

    Oldest experiences are removed when capacity is exceeded.
    Verifiers and aggregator train on this buffer each round.
    """

    def __init__(self, max_size: int = 5000):
        self.max_size = max_size
        self.records: List[SolutionRecord] = []

    def add(self, record: SolutionRecord):
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    def add_batch(self, records: List[SolutionRecord]):
        for r in records:
            self.add(r)

    def get_all(self) -> List[SolutionRecord]:
        return self.records.copy()

    def sample(self, n: int) -> List[SolutionRecord]:
        n = min(n, len(self.records))
        return random.sample(self.records, n)

    def __len__(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Debate state (Irving 2018)
# ---------------------------------------------------------------------------

@dataclass
class DebateMessage:
    verifier_id: int
    round: int
    assessment: str
    confidence: float
    argument: str
    critique: Optional[str] = None


@dataclass
class DebateState:
    problem: str
    solution: str
    messages: List[DebateMessage] = field(default_factory=list)
    verifier_scores: Optional[List[float]] = None
    aggregated_score: Optional[float] = None

    def get_transcript(self, exclude_verifier: Optional[int] = None) -> str:
        if not self.messages:
            return "No previous debate messages."
        parts = []
        for msg in self.messages:
            if exclude_verifier is not None and msg.verifier_id == exclude_verifier:
                continue
            part = (
                f"Verifier {msg.verifier_id} (Round {msg.round}): "
                f"Assessment: {msg.assessment} (confidence: {msg.confidence:.2f})\n"
                f"Argument: {msg.argument}"
            )
            if msg.critique:
                part += f"\nCritique: {msg.critique}"
            parts.append(part)
        return "\n\n".join(parts) if parts else "No other verifier messages yet."

    def get_full_transcript(self) -> str:
        return self.get_transcript(exclude_verifier=None)


# ---------------------------------------------------------------------------
# MathDataset (nvidia/OpenMathInstruct-2)
# ---------------------------------------------------------------------------

class MathDataset:
    """
    Dataset wrapper for OpenMathInstruct-2 from Nvidia.
    Provides sampling and LaTeX-aware answer checking.
    """

    def __init__(self, num_samples: int = 100000):
        self.num_samples = num_samples
        self.dataset = None
        self.train_data = None
        self.test_data = None

    def download(self):
        from datasets import load_dataset
        print("Downloading OpenMathInstruct-2 dataset...")
        self.dataset = load_dataset("nvidia/OpenMathInstruct-2")

        print("Filtering for MATH problems...")
        filtered_data = self.dataset["train_1M"].filter(
            lambda x: x["problem_source"].startswith("math")
        )

        max_samples = min(self.num_samples, len(filtered_data))
        limited_data = filtered_data.select(range(max_samples))
        print(f"Using {max_samples} samples")

        test_size = max_samples // 10
        self.test_data = limited_data.select(range(max_samples - test_size, max_samples))
        self.train_data = limited_data.select(range(max_samples - test_size))
        print(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        return self.dataset

    def get_train_data(self):
        if self.train_data is None:
            self.download()
        return self.train_data

    def get_test_data(self):
        if self.test_data is None:
            self.download()
        return self.test_data

    def sample(self) -> Tuple[str, str]:
        """Returns (problem_text, expected_answer)."""
        if self.train_data is None:
            self.download()
        idx = random.randint(0, len(self.train_data) - 1)
        item = self.train_data[idx]
        return item["problem"], item["expected_answer"]

    def load_math_problems(self, num_problems: int = 100, split: str = "train") -> List[dict]:
        if self.train_data is None:
            self.download()
        data = self.train_data if split == "train" else self.test_data
        problems = []
        for i, item in enumerate(data.select(range(min(num_problems, len(data))))):
            problems.append({
                "problem_id": f"openmath_math_{i}",
                "question": item["problem"],
                "expected_answer": item["expected_answer"],
                "solution_steps": item["generated_solution"],
            })
        return problems

    def check_solution(self, expected_answer: str, predicted_solution: str) -> bool:
        """Check if predicted solution matches expected answer (LaTeX-aware)."""
        try:
            true_val = self._parse_latex_answer(expected_answer)
            if true_val is None:
                return False
            pred_val = self._parse_answer_from_text(predicted_solution)
            if pred_val is None:
                return False
            return self._compare_answers(true_val, pred_val)
        except Exception:
            return False

    def _parse_latex_answer(self, latex_str: str):
        latex_str = latex_str.strip()
        try:
            expr = parse_latex(latex_str)
            if expr.is_number:
                return float(expr)
            if expr.is_Rational:
                return expr
            return expr
        except Exception:
            try:
                clean = latex_str.replace('\\', '').replace('{', '').replace('}', '')
                clean = clean.replace(',', '')
                return float(clean)
            except Exception:
                return latex_str

    def _parse_answer_from_text(self, text: str):
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            return self._parse_latex_answer(boxed_match.group(1))
        answer_match = re.search(r'[Aa]nswer:\s*([^\n]+)', text)
        if answer_match:
            return self._parse_latex_answer(answer_match.group(1))
        gsm_match = re.search(r'####\s*([+-]?\d*\.?\d+)', text)
        if gsm_match:
            return float(gsm_match.group(1))
        final_match = re.search(r'[Ff]inal answer is:?\s*([^\n]+)', text)
        if final_match:
            return self._parse_latex_answer(final_match.group(1))
        lines = text.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and not line.endswith('?'):
                parsed = self._parse_latex_answer(line)
                if parsed is not None:
                    return parsed
        numbers = re.findall(r'([+-]?\d*\.?\d+)', text)
        if numbers:
            try:
                return float(numbers[-1])
            except Exception:
                pass
        return None

    def _compare_answers(self, true_val, pred_val, tolerance=1e-6) -> bool:
        if isinstance(true_val, (int, float)) and isinstance(pred_val, (int, float)):
            return abs(float(true_val) - float(pred_val)) < tolerance
        if hasattr(true_val, 'equals') and hasattr(pred_val, 'equals'):
            try:
                return true_val.equals(pred_val)
            except Exception:
                try:
                    diff = sympy.simplify(true_val - pred_val)
                    return diff == 0
                except Exception:
                    return str(true_val) == str(pred_val)
        try:
            return abs(float(true_val) - float(pred_val)) < tolerance
        except Exception:
            pass
        return str(true_val).strip().lower() == str(pred_val).strip().lower()

    def __len__(self):
        if self.dataset is None:
            self.download()
        return len(self.train_data) + len(self.test_data)
