import torch
import json
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Any
import random
from datetime import datetime

# Import your existing modules
from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
from pmv.aggregator import Aggregator, LearnedAggregator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_models_for_inspection(checkpoint_path, config):
    """Load models from checkpoint for inspection"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    round_idx = checkpoint.get("round", 0)
    print(f"Loaded checkpoint from round {round_idx}")
    
    # Recreate models
    prover_model = config["model"].get("prover_name", "meta-llama/Llama-2-7b-chat-hf")
    verifier_model = config["model"].get("verifier_name", "meta-llama/Llama-2-1b-hf")
    
    prover = Prover(prover_model).to(DEVICE)
    prover.device = DEVICE
    
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = []
    for i in range(num_verifiers):
        try:
            v = Verifier(verifier_model, verifier_type=f"verifier_{i}")
            v.to(DEVICE)
            verifiers.append(v)
        except torch.cuda.OutOfMemoryError:
            print(f"OOM loading verifier {i}, using fewer verifiers")
            break
    
    # Create aggregator
    if config["training"].get("learn_f", False):
        aggregation_type = config["training"].get("aggregation_type", "pl_min")
        aggregator = LearnedAggregator(
            num_verifiers=len(verifiers), 
            aggregation_type=aggregation_type
        ).to(DEVICE)
        
        if "aggregator" in checkpoint:
            aggregator.load_state_dict(checkpoint["aggregator"])
            print("Loaded learned aggregator weights")
    else:
        agg_mode = config["training"].get("aggregator", "softmin")
        aggregator = Aggregator(mode=agg_mode)
    
    return prover, verifiers, aggregator, round_idx

def generate_and_inspect_response(prover, problem, role, max_tokens=1024, temperature=0.7):
    """Generate a response and return detailed information"""
    
    # Create the prompt
    prompt = prover._create_role_prompt(problem, role)
    
    print(f"\n�� GENERATING {role.upper()} RESPONSE")
    print("=" * 60)
    print("PROMPT:")
    print("-" * 20)
    print(prompt)
    print("-" * 20)
    
    # Generate response
    full_response = prover.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
    
    # Extract just the generated part
    if full_response.startswith(prompt):
        generated_text = full_response[len(prompt):]
    else:
        generated_text = full_response
    
    print(f"\nGENERATED RESPONSE ({role}):")
    print("-" * 30)
    print(generated_text)
    print("-" * 30)
    
    return {
        "role": role,
        "problem": problem,
        "prompt": prompt,
        "full_response": full_response,
        "generated_text": generated_text,
        "response_length": len(generated_text),
        "timestamp": datetime.now().isoformat()
    }

def analyze_response_with_verifiers(response_data, verifiers, aggregator, dataset):
    """Analyze a response using all verifiers and aggregator"""
    
    problem = response_data["problem"]
    generated_text = response_data["generated_text"]
    role = response_data["role"]
    
    print(f"\n�� VERIFIER ANALYSIS")
    print("=" * 40)
    
    # Get ground truth
    _, true_solution = dataset.sample_problem_solution(problem) if hasattr(dataset, 'sample_problem_solution') else (problem, "Unknown")
    
    # Check correctness
    try:
        is_correct = dataset.check_solution(true_solution, generated_text)
        print(f"Ground Truth Solution: {true_solution}")
        print(f"Is Correct: {'✅ YES' if is_correct else '❌ NO'}")
    except Exception as e:
        is_correct = None
        print(f"Could not determine correctness: {e}")
    
    print(f"\nRole vs Correctness Analysis:")
    if role == "helpful" and is_correct:
        print("✅ GOOD: Helpful prover gave correct answer")
    elif role == "helpful" and not is_correct:
        print("⚠️  CONCERN: Helpful prover gave incorrect answer")
    elif role == "sneaky" and not is_correct:
        print("✅ EXPECTED: Sneaky prover gave incorrect answer")
    elif role == "sneaky" and is_correct:
        print("⚠️  ISSUE: Sneaky prover gave correct answer (should be sneaky!)")
    
    # Get verifier scores
    print(f"\nVERIFIER SCORES:")
    print("-" * 20)
    verifier_scores = []
    
    for i, verifier in enumerate(verifiers):
        try:
            score = verifier(problem, generated_text)
            verifier_scores.append(score)
            print(f"Verifier {i}: {score:.4f}")
        except Exception as e:
            print(f"Verifier {i}: FAILED ({e})")
            continue
    
    # Aggregate score
    if verifier_scores:
        if isinstance(aggregator, Aggregator):
            agg_score = aggregator(verifier_scores)
            print(f"\nAggregated Score ({aggregator.mode}): {agg_score:.4f}")
        else:
            scores_tensor = torch.tensor([verifier_scores], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                agg_score = aggregator(scores_tensor).item()
            print(f"\nLearned Aggregated Score ({aggregator.aggregation_type}): {agg_score:.4f}")
        
        # Score analysis
        avg_verifier_score = sum(verifier_scores) / len(verifier_scores)
        score_variance = sum((s - avg_verifier_score) ** 2 for s in verifier_scores) / len(verifier_scores)
        
        print(f"Average Verifier Score: {avg_verifier_score:.4f}")
        print(f"Score Variance: {score_variance:.4f}")
        print(f"Score Range: {min(verifier_scores):.4f} - {max(verifier_scores):.4f}")
        
        # Interpretation
        print(f"\nSCORE INTERPRETATION:")
        if role == "helpful":
            if agg_score > 0.7:
                print("✅ GOOD: High score for helpful response")
            elif agg_score < 0.3:
                print("⚠️  CONCERN: Low score for helpful response")
            else:
                print("�� NEUTRAL: Medium score for helpful response")
        else:  # sneaky
            if agg_score > 0.7:
                print("⚠️  SUCCESS (for sneaky): High score despite being sneaky")
            elif agg_score < 0.3:
                print("✅ DETECTED: Low score correctly identifies sneaky response")
            else:
                print("�� PARTIAL: Medium score for sneaky response")
    else:
        print("❌ No verifier scores available")
        agg_score = None
    
    return {
        "verifier_scores": verifier_scores,
        "aggregated_score": agg_score,
        "is_correct": is_correct,
        "score_analysis": {
            "avg_score": sum(verifier_scores) / len(verifier_scores) if verifier_scores else 0,
            "score_variance": sum((s - sum(verifier_scores) / len(verifier_scores)) ** 2 for s in verifier_scores) / len(verifier_scores) if verifier_scores else 0,
            "score_range": [min(verifier_scores), max(verifier_scores)] if verifier_scores else [0, 0]
        }
    }

def compare_helpful_vs_sneaky(prover, verifiers, aggregator, dataset, problem=None, num_comparisons=1):
    """Generate and compare helpful vs sneaky responses for the same problem(s)"""
    
    all_comparisons = []
    
    for i in range(num_comparisons):
        print(f"\n{'='*80}")
        print(f"COMPARISON {i+1}/{num_comparisons}")
        print(f"{'='*80}")
        
        # Get problem
        if problem:
            current_problem = problem
            print(f"Using provided problem: {current_problem[:100]}...")
        else:
            current_problem, _ = dataset.sample()
            print(f"Sampled problem: {current_problem[:100]}...")
        
        # Generate helpful response
        helpful_data = generate_and_inspect_response(prover, current_problem, "helpful")
        helpful_analysis = analyze_response_with_verifiers(helpful_data, verifiers, aggregator, dataset)
        
        print(f"\n{'-'*60}")
        
        # Generate sneaky response
        sneaky_data = generate_and_inspect_response(prover, current_problem, "sneaky")
        sneaky_analysis = analyze_response_with_verifiers(sneaky_data, verifiers, aggregator, dataset)
        
        # Side-by-side comparison
        print(f"\n�� SIDE-BY-SIDE COMPARISON")
        print("=" * 50)
        print(f"{'Metric':<25} {'Helpful':<12} {'Sneaky':<12}")
        print("-" * 50)
        print(f"{'Correctness':<25} {str(helpful_analysis['is_correct']):<12} {str(sneaky_analysis['is_correct']):<12}")
        print(f"{'Avg Verifier Score':<25} {helpful_analysis['score_analysis']['avg_score']:<12.3f} {sneaky_analysis['score_analysis']['avg_score']:<12.3f}")
        print(f"{'Aggregated Score':<25} {helpful_analysis['aggregated_score']:<12.3f} {sneaky_analysis['aggregated_score']:<12.3f}")
        print(f"{'Score Variance':<25} {helpful_analysis['score_analysis']['score_variance']:<12.3f} {sneaky_analysis['score_analysis']['score_variance']:<12.3f}")
        print(f"{'Response Length':<25} {len(helpful_data['generated_text']):<12} {len(sneaky_data['generated_text']):<12}")
        
        # Quality assessment
        print(f"\n�� QUALITY ASSESSMENT:")
        
        # Expected behavior check
        helpful_correct = helpful_analysis['is_correct']
        sneaky_correct = sneaky_analysis['is_correct']
        helpful_score = helpful_analysis['aggregated_score']
        sneaky_score = sneaky_analysis['aggregated_score']
        
        issues = []
        successes = []
        
        if helpful_correct:
            successes.append("✅ Helpful prover gave correct answer")
        else:
            issues.append("⚠️  Helpful prover gave incorrect answer")
        
        if not sneaky_correct:
            successes.append("✅ Sneaky prover gave incorrect answer (as expected)")
        else:
            issues.append("⚠️  Sneaky prover gave correct answer (not sneaky enough!)")
        
        if helpful_score and sneaky_score:
            if helpful_score > sneaky_score:
                successes.append("✅ Helpful scored higher than sneaky")
            else:
                issues.append("⚠️  Sneaky scored higher than or equal to helpful")
        
        print("SUCCESSES:")
        for success in successes:
            print(f"  {success}")
        
        if issues:
            print("ISSUES:")
            for issue in issues:
                print(f"  {issue}")
        
        comparison_data = {
            "problem": current_problem,
            "helpful": {**helpful_data, **helpful_analysis},
            "sneaky": {**sneaky_data, **sneaky_analysis},
            "comparison_summary": {
                "successes": successes,
                "issues": issues,
                "helpful_better": helpful_score > sneaky_score if (helpful_score and sneaky_score) else None
            }
        }
        
        all_comparisons.append(comparison_data)
    
    return all_comparisons

def inspect_replay_buffer_samples(checkpoint_path, num_samples=5):
    """Inspect some samples from the replay buffer"""
    
    print(f"\n��️  REPLAY BUFFER INSPECTION")
    print("=" * 50)
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    replay_buffer = checkpoint.get("replay_buffer", [])
    
    if not replay_buffer:
        print("No replay buffer found in checkpoint")
        return
    
    print(f"Replay buffer contains {len(replay_buffer)} experiences")
    
    # Sample random experiences
    sample_indices = random.sample(range(len(replay_buffer)), min(num_samples, len(replay_buffer)))
    
    for i, idx in enumerate(sample_indices):
        experience = replay_buffer[idx]
        problem, true_solution, response, reward, _, role = experience
        
        print(f"\n--- REPLAY SAMPLE {i+1} (Index {idx}) ---")
        print(f"Role: {role}")
        print(f"Reward: {reward:.4f}")
        print(f"Problem: {problem[:100]}...")
        print(f"True Solution: {true_solution}")
        print(f"Generated Response: {response[:200]}...")
        print(f"Response Length: {len(response)} chars")

def interactive_mode(prover, verifiers, aggregator, dataset):
    """Interactive mode for exploring responses"""
    
    print(f"\n�� INTERACTIVE MODE")
    print("=" * 40)
    print("Commands:")
    print("  'sample' - Generate responses for a random problem")
    print("  'problem <text>' - Generate responses for a specific problem")
    print("  'compare' - Compare helpful vs sneaky for random problem")
    print("  'quit' - Exit interactive mode")
    print()
    
    while True:
        try:
            cmd = input(">>> ").strip()
            
            if cmd == 'quit':
                break
            elif cmd == 'sample':
                problem, _ = dataset.sample()
                print(f"\nSampled problem: {problem}")
                role = input("Role (helpful/sneaky): ").strip()
                if role in ['helpful', 'sneaky']:
                    response_data = generate_and_inspect_response(prover, problem, role)
                    analyze_response_with_verifiers(response_data, verifiers, aggregator, dataset)
                else:
                    print("Invalid role. Use 'helpful' or 'sneaky'")
            elif cmd.startswith('problem '):
                problem = cmd[8:].strip()
                if problem:
                    role = input("Role (helpful/sneaky): ").strip()
                    if role in ['helpful', 'sneaky']:
                        response_data = generate_and_inspect_response(prover, problem, role)
                        analyze_response_with_verifiers(response_data, verifiers, aggregator, dataset)
                    else:
                        print("Invalid role. Use 'helpful' or 'sneaky'")
                else:
                    print("No problem provided")
            elif cmd == 'compare':
                compare_helpful_vs_sneaky(prover, verifiers, aggregator, dataset, num_comparisons=1)
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")

def save_inspection_results(results, output_path):
    """Save inspection results to JSON file"""
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Inspect model outputs and reasoning traces")
    parser.add_argument("checkpoint_path", help="Path to the checkpoint file (.pt)")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--mode", choices=["compare", "interactive", "replay", "single"], 
                       default="compare", help="Inspection mode")
    parser.add_argument("--problem", help="Specific problem to test (for single mode)")
    parser.add_argument("--role", choices=["helpful", "sneaky"], help="Role for single mode")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of comparisons/samples")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = args.config
    else:
        checkpoint_dir = Path(args.checkpoint_path).parent
        config_path = checkpoint_dir / "config.yaml"
        if not config_path.exists():
            print(f"Config not found at {config_path}, please specify --config")
            return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load models
    dataset = MathDataset()
    prover, verifiers, aggregator, round_idx = load_models_for_inspection(args.checkpoint_path, config)
    
    print(f"\nLoaded models from round {round_idx}")
    print(f"Mode: {args.mode}")
    
    # Run based on mode
    if args.mode == "compare":
        results = compare_helpful_vs_sneaky(
            prover, verifiers, aggregator, dataset, 
            problem=args.problem, num_comparisons=args.num_samples
        )
        if args.output:
            save_inspection_results(results, args.output)
            
    elif args.mode == "single":
        if not args.problem:
            problem, _ = dataset.sample()
        else:
            problem = args.problem
            
        role = args.role or "helpful"
        
        response_data = generate_and_inspect_response(prover, problem, role, temperature=args.temperature)
        analysis = analyze_response_with_verifiers(response_data, verifiers, aggregator, dataset)
        
        if args.output:
            results = {**response_data, **analysis}
            save_inspection_results(results, args.output)
            
    elif args.mode == "replay":
        inspect_replay_buffer_samples(args.checkpoint_path, args.num_samples)
        
    elif args.mode == "interactive":
        interactive_mode(prover, verifiers, aggregator, dataset)

if __name__ == "__main__":
    main()





# # Compare 3 pairs of helpful vs sneaky responses
# python3 inspect_traces.py runs/pure_stackelberg_experiment_20250729_124028/checkpoints/kirchner_round_013.pt --mode compare --num_samples 3 --config runs/pure_stackelberg_experiment_20250729_124028/config.yaml

# # Compare for a specific problem
# python inspect_traces.py checkpoint.pt --mode compare --problem "Solve for x: 2x + 5 = 13"

# # Generate and analyze a helpful response
# python inspect_traces.py checkpoint.pt --mode single --role helpful

# # Generate sneaky response for specific problem
# python inspect_traces.py checkpoint.pt --mode single --role sneaky --problem "What is 2+2?"

# # Interactive exploration
# python inspect_traces.py checkpoint.pt --mode interactive

# # Look at stored experiences from training
# python inspect_traces.py checkpoint.pt --mode replay --num_samples 5


# # Save detailed traces to JSON
# python inspect_traces.py checkpoint.pt --mode compare --num_samples 5 --output traces_round_5.json