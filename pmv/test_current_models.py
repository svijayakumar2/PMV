# test_current_models.py - FIXED VERSION
import torch
import yaml
from models.prover import Prover
from models.verifier import Verifier
from data.math_dataset import MathDataset
from peft import PeftModel

def test_models():
    # Load your trained checkpoint
    checkpoint = torch.load("runs/pure_stackelberg_experiment_test/checkpoints/pure_stackelberg_round_001.pt")
    
    # Load config
    config = checkpoint["config"]
    
    # Initialize base models (without LoRA)
    model_name = config["model"]["name"]
    prover = Prover(model_name)
    
    # Load LoRA adapters instead of trying to load full state dict
    if config["training"].get("use_lora", True):
        # Load from the saved LoRA directory
        lora_path = "runs/pure_stackelberg_experiment_test/checkpoints/lora_round_001"
        prover.model = PeftModel.from_pretrained(prover.model, lora_path)
        print("Loaded LoRA adapters for prover")
    else:
        # If not using LoRA, load normally
        prover.load_state_dict(checkpoint["prover"])
    
    # Load verifiers (these should work normally since they're smaller)
    verifiers = []
    for i, state_dict in enumerate(checkpoint["verifiers"]):
        verifier = Verifier(model_name, verifier_type=f"verifier_{i}")
        verifier.load_state_dict(state_dict)
        verifiers.append(verifier)
    
    # Test on some problems
    dataset = MathDataset()
    
    for i in range(3):
        problem, true_solution = dataset.sample()
        print(f"\n=== Problem {i+1} ===")
        print(f"Problem: {problem}")
        
        # Test helpful prover
        helpful_prompt = prover._create_role_prompt(problem, "helpful")
        helpful_response = prover.generate(helpful_prompt, max_new_tokens=256)
        helpful_solution = helpful_response[len(helpful_prompt):] if helpful_response.startswith(helpful_prompt) else helpful_response
        
        # Test sneaky prover  
        sneaky_prompt = prover._create_role_prompt(problem, "sneaky")
        sneaky_response = prover.generate(sneaky_prompt, max_new_tokens=256)
        sneaky_solution = sneaky_response[len(sneaky_prompt):] if sneaky_response.startswith(sneaky_prompt) else sneaky_response
        
        # Get verifier scores
        helpful_scores = [v(problem, helpful_solution) for v in verifiers]
        sneaky_scores = [v(problem, sneaky_solution) for v in verifiers]
        
        print(f"\nHelpful Solution: {helpful_solution[:200]}...")
        print(f"Helpful Scores: {helpful_scores} (avg: {sum(helpful_scores)/len(helpful_scores):.3f})")
        
        print(f"\nSneaky Solution: {sneaky_solution[:200]}...")
        print(f"Sneaky Scores: {sneaky_scores} (avg: {sum(sneaky_scores)/len(sneaky_scores):.3f})")

if __name__ == "__main__":
    test_models()
