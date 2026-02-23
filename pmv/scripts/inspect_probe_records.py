#!/usr/bin/env python3
"""Quick inspector for evaluation JSON probe_records."""

import argparse
import json


def _short(text: str, n: int = 220) -> str:
    t = (text or "").replace("\r\n", "\n").strip()
    if len(t) <= n:
        return t
    return t[: n - 3] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_json", help="Path to eval JSON with probe_records")
    parser.add_argument("--show", type=int, default=5, help="How many helpful-failure examples to print")
    args = parser.parse_args()

    with open(args.eval_json) as f:
        data = json.load(f)

    records = data.get("probe_records", [])
    if not records:
        print("No probe_records found. Re-run evaluation with --save-probe-records.")
        return

    n = len(records)
    had_final = sum(1 for r in records if r.get("had_final_before_fix"))
    raw_correct = sum(1 for r in records if r.get("correct_raw"))
    final_correct = sum(1 for r in records if r.get("correct_final"))
    changed = sum(
        1
        for r in records
        if bool(r.get("correct_raw")) != bool(r.get("correct_final"))
    )

    helpful = [r for r in records if r.get("role") == "helpful"]
    helpful_bad = [r for r in helpful if not r.get("correct_final")]

    print(f"episodes: {n}")
    print(f"had FINAL before fix: {had_final}/{n} ({had_final/max(n,1):.1%})")
    print(f"correct raw: {raw_correct}/{n} ({raw_correct/max(n,1):.1%})")
    print(f"correct final: {final_correct}/{n} ({final_correct/max(n,1):.1%})")
    print(f"correctness changed by final-fix: {changed}/{n} ({changed/max(n,1):.1%})")
    print(f"helpful incorrect: {len(helpful_bad)}/{max(len(helpful),1)} ({len(helpful_bad)/max(len(helpful),1):.1%})")

    if not helpful_bad:
        return

    print("\nExamples: helpful incorrect")
    for rec in helpful_bad[: max(0, args.show)]:
        print("-" * 80)
        print(f"episode: {rec.get('episode')}, oversight_score={rec.get('oversight_score')}")
        print(f"solution_true: {_short(rec.get('solution_true', ''))}")
        print(f"response_raw: {_short(rec.get('response_raw', ''))}")
        print(f"response_final: {_short(rec.get('response_final', ''))}")


if __name__ == "__main__":
    main()
