# test_stackelberg_models.py
import torch
import yaml
from models.prover import Prover
from models.verifier import Verifier
from data.math_dataset import MathDataset

def test_models():
    # Load your trained checkpoint
    checkpoint = torch.load("runs/pure_stackelberg_experiment/checkpoints/pure_stackelberg_round_014.pt")
    
    # Load config
    config = checkpoint["config"]
    
    # Initialize models
    model_name = config["model"]["name"]
    prover = Prover(model_name)
    prover.load_state_dict(checkpoint["prover"])
    
    verifiers = []
    for i, state_dict in enumerate(checkpoint["verifiers"]):
        verifier = Verifier(model_name, verifier_type=f"verifier_{i}")
        verifier.load_state_dict(state_dict)
        verifiers.append(verifier)
    
    # Test on some problems
    dataset = MathDataset()
    
    for i in range(5):
        problem, true_solution = dataset.sample()
        print(f"\n=== Problem {i+1} ===")
        print(f"Problem: {problem}")
        
        # Test helpful prover
        helpful_prompt = prover._create_role_prompt(problem, "helpful")
        helpful_response = prover.generate(helpful_prompt, max_new_tokens=256)
        helpful_solution = helpful_response[len(helpful_prompt):]
        
        # Test sneaky prover  
        sneaky_prompt = prover._create_role_prompt(problem, "sneaky")
        sneaky_response = prover.generate(sneaky_prompt, max_new_tokens=256)
        sneaky_solution = sneaky_response[len(sneaky_prompt):]
        
        # Get verifier scores
        helpful_scores = [v(problem, helpful_solution) for v in verifiers]
        sneaky_scores = [v(problem, sneaky_solution) for v in verifiers]
        
        print(f"\nHelpful Solution: {helpful_solution[:200]}...")
        print(f"Helpful Scores: {helpful_scores} (avg: {sum(helpful_scores)/len(helpful_scores):.3f})")
        
        print(f"\nSneaky Solution: {sneaky_solution[:200]}...")
        print(f"Sneaky Scores: {sneaky_scores} (avg: {sum(sneaky_scores)/len(sneaky_scores):.3f})")
        
        print(f"True Solution: {true_solution[:200]}...")

if __name__ == "__main__":
    test_models()
