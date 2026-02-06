"""
ZebraLogic dataset adapter for PMV games.

Logic grid puzzles (constraint satisfaction problems) from:
  Lin et al. 2025, "ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning"
  https://huggingface.co/datasets/WildEval/ZebraLogic

Each puzzle: N houses × M features, a set of natural-language clues,
and a unique solution grid. The solution is a JSON dict with "header"
and "rows". Puzzles have controllable complexity via search space size
and Z3 conflict count.

This domain tests verifier diversity on a fundamentally different axis
than math: logical constraint propagation, non-monotonic reasoning,
and compositional generalization. The "curse of complexity" finding
from Lin et al. makes it a good stress test for oversight.
"""

import json
import random
import re
from typing import Tuple, List, Dict, Optional


class ZebraLogicDataset:
    """
    Wraps HuggingFace WildEval/ZebraLogic in the same interface as
    DummyMathDataset: sample() -> (problem, solution_true) and
    check_solution(solution_true, response) -> bool.

    Loads from HuggingFace datasets library. Falls back to a small
    built-in set if the library is unavailable.
    """

    def __init__(
        self,
        subset: str = "grid_mode",
        split: str = "test",
        max_size: Optional[str] = None,
        min_size: Optional[str] = None,
    ):
        """
        Args:
            subset: "grid_mode" (free-form answer) or "mc_mode" (multiple choice).
            split: dataset split.
            max_size: filter puzzles, e.g. "4*4" means at most 4 houses × 4 features.
            min_size: filter puzzles, e.g. "3*3" means at least 3×3.
        """
        self.puzzles: List[Dict] = []
        try:
            from datasets import load_dataset
            ds = load_dataset("WildEval/ZebraLogic", subset, split=split)
            for row in ds:
                puzzle_entry = {
                    "id": row["id"],
                    "size": row["size"],
                    "puzzle_text": row["puzzle"],
                    "solution": row["solution"] if isinstance(row["solution"], dict)
                                else json.loads(row["solution"]),
                }
                if max_size and not self._size_leq(row["size"], max_size):
                    continue
                if min_size and not self._size_geq(row["size"], min_size):
                    continue
                self.puzzles.append(puzzle_entry)
        except Exception as e:
            print(f"Warning: Could not load ZebraLogic from HuggingFace ({e}). "
                  f"Using built-in fallback puzzles.")
            self.puzzles = self._fallback_puzzles()

        print(f"ZebraLogicDataset: loaded {len(self.puzzles)} puzzles")

    @staticmethod
    def _parse_size(size_str: str) -> Tuple[int, int]:
        parts = size_str.split("*")
        return int(parts[0]), int(parts[1])

    def _size_leq(self, size_str: str, max_str: str) -> bool:
        h, f = self._parse_size(size_str)
        mh, mf = self._parse_size(max_str)
        return h <= mh and f <= mf

    def _size_geq(self, size_str: str, min_str: str) -> bool:
        h, f = self._parse_size(size_str)
        mh, mf = self._parse_size(min_str)
        return h >= mh and f >= mf

    def sample(self) -> Tuple[str, str]:
        """Returns (puzzle_text, solution_json_string)."""
        entry = random.choice(self.puzzles)
        return entry["puzzle_text"], json.dumps(entry["solution"])

    def sample_with_metadata(self) -> Dict:
        return random.choice(self.puzzles)

    def check_solution(self, correct_answer: str, response: str) -> bool:
        """
        Check if the response contains the correct grid assignment.

        Tries multiple strategies:
          1. Parse a JSON grid from the response and compare cell-by-cell.
          2. Check if all solution rows appear as substrings.
          3. Check if all (house, attribute, value) triples are present.
        """
        try:
            solution = json.loads(correct_answer)
        except (json.JSONDecodeError, TypeError):
            return False

        rows = solution.get("rows", [])
        header = solution.get("header", [])

        # Strategy 1: try to parse a JSON grid from the response
        parsed = self._try_parse_grid(response)
        if parsed is not None:
            return self._grids_match(rows, parsed)

        # Strategy 2: check if every (header[col], value) pair for each row
        # appears in the response text
        response_lower = response.lower()
        all_found = True
        for row in rows:
            house_num = row[0]
            for col_idx in range(1, min(len(header), len(row))):
                attr = header[col_idx].lower()
                val = row[col_idx].lower()
                # Look for "house N ... value" or "N: value" patterns
                if val not in response_lower:
                    all_found = False
                    break
            if not all_found:
                break

        if all_found:
            return True

        # Strategy 3: check that every value from the solution grid appears
        # in the correct position context
        match_count = 0
        total = 0
        for row in rows:
            house = row[0]
            for col_idx in range(1, len(row)):
                total += 1
                val = row[col_idx].lower()
                if val in response_lower:
                    match_count += 1

        return total > 0 and match_count == total

    @staticmethod
    def _try_parse_grid(response: str) -> Optional[List[List[str]]]:
        """Try to extract a JSON grid from the response."""
        # Look for JSON blocks
        json_patterns = [
            re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL),
            re.compile(r'(\{[^{}]*"rows"\s*:\s*\[.*?\]\s*[^{}]*\})', re.DOTALL),
            re.compile(r'(\{[^{}]*"header"\s*:\s*\[.*?\].*?\})', re.DOTALL),
        ]
        for pat in json_patterns:
            match = pat.search(response)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    if "rows" in parsed:
                        return parsed["rows"]
                except (json.JSONDecodeError, KeyError):
                    continue
        return None

    @staticmethod
    def _grids_match(expected: List[List[str]], actual: List[List[str]]) -> bool:
        if len(expected) != len(actual):
            return False
        for exp_row, act_row in zip(expected, actual):
            if len(exp_row) != len(act_row):
                return False
            for e, a in zip(exp_row, act_row):
                if str(e).strip().lower() != str(a).strip().lower():
                    return False
        return True

    @staticmethod
    def _fallback_puzzles() -> List[Dict]:
        """Minimal built-in puzzles for testing without HuggingFace."""
        return [
            {
                "id": "fallback-2x3-1",
                "size": "2*3",
                "puzzle_text": (
                    "There are 2 houses, numbered 1 to 2 from left to right.\n"
                    "Each house has unique attributes:\n"
                    "- Names: Alice, Bob\n"
                    "- Colors: red, blue\n"
                    "- Pets: cat, dog\n\n"
                    "Clues:\n"
                    "1. Alice lives in house 1.\n"
                    "2. The person in the red house has a cat.\n"
                    "3. Bob's house is blue.\n"
                ),
                "solution": {
                    "header": ["House", "Name", "Color", "Pet"],
                    "rows": [
                        ["1", "Alice", "red", "cat"],
                        ["2", "Bob", "blue", "dog"],
                    ]
                },
            },
            {
                "id": "fallback-3x3-1",
                "size": "3*3",
                "puzzle_text": (
                    "There are 3 houses, numbered 1 to 3 from left to right.\n"
                    "Each house has unique attributes:\n"
                    "- Names: Alice, Bob, Carol\n"
                    "- Drinks: water, tea, coffee\n"
                    "- Pets: cat, dog, fish\n\n"
                    "Clues:\n"
                    "1. Alice lives in house 1.\n"
                    "2. The tea drinker has a dog.\n"
                    "3. Carol lives in house 3.\n"
                    "4. The person in house 2 drinks coffee.\n"
                    "5. Alice has a cat.\n"
                ),
                "solution": {
                    "header": ["House", "Name", "Drink", "Pet"],
                    "rows": [
                        ["1", "Alice", "water", "cat"],
                        ["2", "Bob", "coffee", "fish"],
                        ["3", "Carol", "tea", "dog"],
                    ]
                },
            },
        ]
