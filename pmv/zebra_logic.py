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

    def sample_train(self) -> Tuple[str, str]:
        """Compatibility shim: Zebra is loaded as a fixed split at init."""
        return self.sample()

    def sample_eval(self) -> Tuple[str, str]:
        """Compatibility shim: Zebra is loaded as a fixed split at init."""
        return self.sample()

    def sample_with_metadata(self) -> Dict:
        return random.choice(self.puzzles)

    def check_solution(self, correct_answer: str, response: str) -> bool:
        """
        Check if the response contains the correct full grid assignment.

        The matcher is intentionally strict to reduce false positives:
          1. Prefer structured outputs (JSON grid / table / house lines).
          2. Fallback only to house-conditioned textual assignment checks.
          3. Never accept by global value presence alone.
        """
        try:
            solution = json.loads(correct_answer)
        except (json.JSONDecodeError, TypeError):
            return False

        rows = solution.get("rows", [])
        if not rows:
            return False

        expected_cols = len(rows[0])
        final_payload = self._extract_final_payload(response)
        candidates = [final_payload, response] if final_payload else [response]

        # Strategy 1: parse structured outputs and compare exact assignments.
        for text in candidates:
            parsed = self._try_parse_grid(text)
            if parsed is not None and self._grids_match(rows, parsed):
                return True

            parsed_table = self._try_parse_table(text, expected_cols=expected_cols)
            if parsed_table is not None and self._grids_match(rows, parsed_table):
                return True

            parsed_house_lines = self._try_parse_house_lines(text, expected_cols=expected_cols)
            if parsed_house_lines is not None and self._grids_match(rows, parsed_house_lines):
                return True

        # Strategy 2: strict textual fallback requiring each value to be linked
        # to the correct house in local context.
        for text in candidates:
            if self._matches_house_assignments(rows, text):
                return True

        return False

    @staticmethod
    def _try_parse_grid(response: str) -> Optional[List[List[str]]]:
        """Try to extract a JSON grid from the response."""
        # Look for JSON blocks
        json_patterns = [
            re.compile(r'```json\s*(\[[\s\S]*?\])\s*```', re.IGNORECASE),
            re.compile(r'```json\s*(\{.*?\})\s*```', re.DOTALL),
            re.compile(r'(\[[\s\S]*\])', re.DOTALL),
            re.compile(r'(\{[^{}]*"rows"\s*:\s*\[.*?\]\s*[^{}]*\})', re.DOTALL),
            re.compile(r'(\{[^{}]*"header"\s*:\s*\[.*?\].*?\})', re.DOTALL),
        ]
        for pat in json_patterns:
            match = pat.search(response)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    rows = ZebraLogicDataset._coerce_rows(parsed)
                    if rows is not None:
                        return rows
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _extract_final_payload(response: str) -> Optional[str]:
        for line in reversed(response.splitlines()):
            m = re.match(r"^\s*FINAL:\s*(.+)\s*$", line, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return None

    @staticmethod
    def _coerce_rows(parsed_obj) -> Optional[List[List[str]]]:
        rows_obj = None
        if isinstance(parsed_obj, dict):
            if "rows" in parsed_obj:
                rows_obj = parsed_obj["rows"]
            elif "solution" in parsed_obj and isinstance(parsed_obj["solution"], dict):
                rows_obj = parsed_obj["solution"].get("rows")
        elif isinstance(parsed_obj, list):
            rows_obj = parsed_obj
        if not isinstance(rows_obj, list):
            return None
        rows: List[List[str]] = []
        for row in rows_obj:
            if not isinstance(row, list):
                return None
            rows.append([str(cell).strip() for cell in row])
        return rows if rows else None

    @staticmethod
    def _try_parse_table(response: str, expected_cols: int) -> Optional[List[List[str]]]:
        rows: List[List[str]] = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if "|" not in line:
                continue
            line = line.strip("|")
            cells = [c.strip() for c in line.split("|")]
            if len(cells) < expected_cols:
                continue
            # Skip markdown separator rows like |---|---|
            if all(re.fullmatch(r"[:\- ]+", c or "-") for c in cells):
                continue
            # Require a numeric house id in first column to avoid header lines.
            if not re.fullmatch(r"\d+", cells[0]):
                continue
            rows.append(cells[:expected_cols])
        return rows if rows else None

    @staticmethod
    def _try_parse_house_lines(response: str, expected_cols: int) -> Optional[List[List[str]]]:
        rows: List[List[str]] = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            m = re.match(r"^(?:house\s*)?(\d+)\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
            if not m:
                continue
            house = m.group(1).strip()
            rhs = m.group(2).strip()
            parts = [p.strip() for p in re.split(r"[,\|;]", rhs) if p.strip()]
            if len(parts) < expected_cols - 1:
                continue
            rows.append([house] + parts[: expected_cols - 1])
        return rows if rows else None

    @staticmethod
    def _matches_house_assignments(expected_rows: List[List[str]], response: str) -> bool:
        """
        Strict textual fallback: every expected value must appear near the
        correct house id on at least one line.
        """
        text = response.lower()
        for row in expected_rows:
            house = re.escape(str(row[0]).strip().lower())
            for cell in row[1:]:
                val = re.escape(str(cell).strip().lower())
                line_level = re.search(
                    rf"(^|[\n\r]).*(house\s*{house}\b.*\b{val}\b|\b{val}\b.*house\s*{house}\b).*($|[\n\r])",
                    text,
                )
                if line_level:
                    continue
                local_window = re.search(
                    rf"(house\s*{house}\b.{0,80}\b{val}\b|\b{val}\b.{0,80}house\s*{house}\b)",
                    text,
                    flags=re.DOTALL,
                )
                if not local_window:
                    return False
        return True

    @staticmethod
    def _grids_match(expected: List[List[str]], actual: List[List[str]]) -> bool:
        if len(expected) != len(actual):
            return False
        expected_map = {}
        for exp_row in expected:
            if len(exp_row) < 2:
                return False
            key = str(exp_row[0]).strip().lower()
            expected_map[key] = [str(c).strip().lower() for c in exp_row[1:]]
        actual_map = {}
        for act_row in actual:
            if len(act_row) < 2:
                return False
            key = str(act_row[0]).strip().lower()
            actual_map[key] = [str(c).strip().lower() for c in act_row[1:]]
        if set(expected_map.keys()) != set(actual_map.keys()):
            return False
        for key in expected_map:
            if len(expected_map[key]) != len(actual_map[key]):
                return False
            for e, a in zip(expected_map[key], actual_map[key]):
                if e != a:
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
