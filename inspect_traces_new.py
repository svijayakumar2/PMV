import torch
import yaml
import json
from pathlib import Path
from typing import List, Dict
import os

from pmv.data.math_dataset import MathDataset
from pmv.models.prover import Prover
from pmv.models.verifier import Verifier
from pmv.aggregator import Aggregator, LearnedAggregator
from peft import LoraConfig, get_peft_model, TaskType

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set cache directories
os.environ['HF_HOME'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/dccstor/principled_ai/users/saranyaibm2/hf_cache'


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def setup_lora(model, config):
    """Setup LoRA for parameter-efficient fine-tuning."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config["training"].get("lora_r", 8),
        lora_alpha=config["training"].get("lora_alpha", 32),
        lora_dropout=config["training"].get("lora_dropout", 0.1),
        target_modules=config["training"].get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    )
    return get_peft_model(model, lora_config)


def load_models_from_checkpoint(checkpoint_path: str):
    """Load models and config from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load config
    config = checkpoint.get("config")
    if config is None:
        # Try to load from the checkpoint directory
        ckpt_dir = Path(checkpoint_path).parent.parent
        config_path = ckpt_dir / "config.yaml"
        if config_path.exists():
            config = load_config(config_path)
        else:
            raise ValueError("Config not found in checkpoint or directory")
    
    # Initialize prover
    prover_model = config["model"].get("prover_name", "meta-llama/Llama-2-7b-chat-hf")
    prover = Prover(prover_model).to(DEVICE)
    prover.config = config
    
    if config["training"].get("use_lora", True):
        prover.model = setup_lora(prover.model, config)
    
    prover.device = DEVICE
    
    # Initialize verifiers
    verifier_model = config["model"].get("verifier_name", "meta-llama/Llama-2-1b-hf")
    num_verifiers = config["model"].get("num_verifiers", 3)
    verifiers = []
    
    for i in range(num_verifiers):
        v = Verifier(verifier_model, verifier_type=f"verifier_{i}")
        v.config = config
        v.to(DEVICE)
        verifiers.append(v)
    
    # Initialize aggregator
    if config["training"].get("learn_f", False):
        aggregation_type = config["training"].get("aggregation_type", "pl_min")
        aggregator = LearnedAggregator(
            num_verifiers=num_verifiers,
            aggregation_type=aggregation_type
        ).to(DEVICE)
        
        if "aggregator" in checkpoint:
            aggregator.load_state_dict(checkpoint["aggregator"])
            print("Loaded aggregator state")
    else:
        agg_mode = config["training"].get("aggregator", "softmin")
        aggregator = Aggregator(mode=agg_mode)
    
    round_num = checkpoint.get("round", 0)
    print(f"Loaded checkpoint from round {round_num}")
    
    return prover, verifiers, aggregator, config, round_num


def generate_and_inspect_traces(
    prover: Prover,
    verifiers: List[Verifier],
    aggregator,
    dataset: MathDataset,
    num_samples: int = 10,
    output_file: str = "reasoning_traces.json"
):
    """Generate reasoning traces and inspect them"""
    
    results = []
    
    for i in range(num_samples):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        # Get a problem
        problem, solution_true = dataset.sample()
        
        # Generate helpful response
        print("\n--- HELPFUL RESPONSE ---")
        helpful_prompt = prover._create_role_prompt(problem, "helpful")
        helpful_response = prover.generate(helpful_prompt, max_new_tokens=2000)
        helpful_text = helpful_response[len(helpful_prompt):] if helpful_response.startswith(helpful_prompt) else helpful_response
        
        # Check correctness
        helpful_correct = dataset.check_solution(solution_true, helpful_text)
        
        # Get verifier scores
        helpful_scores = []
        for v_idx, verifier in enumerate(verifiers):
            try:
                score = verifier(problem, helpful_text)
                helpful_scores.append(score)
            except Exception as e:
                print(f"Verifier {v_idx} failed: {e}")
                helpful_scores.append(-15.0)
        
        # Aggregate score
        if isinstance(aggregator, Aggregator):
            helpful_agg_score = aggregator(helpful_scores)
        else:
            scores_tensor = torch.tensor(helpful_scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            helpful_agg_score = aggregator(scores_tensor).item()
        
        print(f"Problem: {problem}")
        print(f"True Solution: {solution_true}")
        print(f"Helpful Response: {helpful_text}")
        print(f"Helpful Correct: {helpful_correct}")
        print(f"Helpful Verifier Scores: {helpful_scores}")
        print(f"Helpful Aggregated Score: {helpful_agg_score:.4f}")
        
        # Generate sneaky response
        print("\n--- SNEAKY RESPONSE ---")
        sneaky_prompt = prover._create_role_prompt(problem, "sneaky")
        sneaky_response = prover.generate(sneaky_prompt, max_new_tokens=2000)
        sneaky_text = sneaky_response[len(sneaky_prompt):] if sneaky_response.startswith(sneaky_prompt) else sneaky_response
        
        # Check correctness
        sneaky_correct = dataset.check_solution(solution_true, sneaky_text)
        
        # Get verifier scores
        sneaky_scores = []
        for v_idx, verifier in enumerate(verifiers):
            try:
                score = verifier(problem, sneaky_text)
                sneaky_scores.append(score)
            except Exception as e:
                print(f"Verifier {v_idx} failed: {e}")
                sneaky_scores.append(0.0)
        
        # Aggregate score
        if isinstance(aggregator, Aggregator):
            sneaky_agg_score = aggregator(sneaky_scores)
        else:
            scores_tensor = torch.tensor(sneaky_scores, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            sneaky_agg_score = aggregator(scores_tensor).item()
        
        print(f"Sneaky Response: {sneaky_text}")
        print(f"Sneaky Correct: {sneaky_correct}")
        print(f"Sneaky Verifier Scores: {sneaky_scores}")
        print(f"Sneaky Aggregated Score: {sneaky_agg_score:.4f}")
        
        # Store results
        result = {
            "sample_id": i,
            "problem": problem,
            "true_solution": solution_true,
            "helpful": {
                "response": helpful_text,
                "correct": helpful_correct,
                "verifier_scores": helpful_scores,
                "aggregated_score": helpful_agg_score
            },
            "sneaky": {
                "response": sneaky_text,
                "correct": sneaky_correct,
                "verifier_scores": sneaky_scores,
                "aggregated_score": sneaky_agg_score
            }
        }
        results.append(result)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    helpful_correct_count = sum(1 for r in results if r["helpful"]["correct"])
    sneaky_correct_count = sum(1 for r in results if r["sneaky"]["correct"])
    
    helpful_avg_score = sum(r["helpful"]["aggregated_score"] for r in results) / len(results)
    sneaky_avg_score = sum(r["sneaky"]["aggregated_score"] for r in results) / len(results)
    
    print(f"Helpful correct rate: {helpful_correct_count}/{num_samples} ({100*helpful_correct_count/num_samples:.1f}%)")
    print(f"Sneaky correct rate: {sneaky_correct_count}/{num_samples} ({100*sneaky_correct_count/num_samples:.1f}%)")
    print(f"Helpful avg verifier score: {helpful_avg_score:.4f}")
    print(f"Sneaky avg verifier score: {sneaky_avg_score:.4f}")
    
    # Check if sneaky is fooling the verifier
    sneaky_incorrect_high_score = sum(1 for r in results 
                                      if not r["sneaky"]["correct"] 
                                      and r["sneaky"]["aggregated_score"] > 0.7)
    print(f"Sneaky incorrect with high score (>0.7): {sneaky_incorrect_high_score}/{num_samples}")


def main():
    # Configuration
    checkpoint_path = "runs/pure_stackelberg_experiment_20250807_095153/checkpoints/kirchner_round_003.pt"
    num_samples = 20
    output_file = "reasoning_traces_analysis3rdcheckpoint.json"
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path variable with the correct path")
        return
    
    # Load models
    prover, verifiers, aggregator, config, round_num = load_models_from_checkpoint(checkpoint_path)
    
    # Initialize dataset
    dataset = MathDataset()
    
    # Generate and inspect traces
    generate_and_inspect_traces(
        prover=prover,
        verifiers=verifiers,
        aggregator=aggregator,
        dataset=dataset,
        num_samples=num_samples,
        output_file=output_file
    )


if __name__ == "__main__":
    main()
