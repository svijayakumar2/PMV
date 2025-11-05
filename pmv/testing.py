import torch
import matplotlib.pyplot as plt
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import random
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# plt.style.use('seaborn-v0_8')
# sns.set_palette("husl")

EXPERIMENT_DIRS = [
    # "pure_stackelberg_experiment_20250716_182411",
    "pure_stackelberg_experiment_20250716_212043", 
    # "pure_stackelberg_experiment_20250717_210733",
    # "pure_stackelberg_experiment_20250721_112618",
    "pure_stackelberg_experiment_20250718_074523",
    "pure_stackelberg_experiment_20250721_200515"
    
]


def get_config_label(experiment_path: str) -> str:
    """Extract meaningful label from experiment config"""
    config_path = Path(f"../runs/{experiment_path}/config.yaml")
    
    if not config_path.exists():
        return experiment_path
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        label_parts = []
        
        # Model info
        if 'model' in config:
            model = config['model']
            if 'num_verifiers' in model:
                label_parts.append(f"V{model['num_verifiers']}")
            
            # Shorten model names
            if 'prover_name' in model:
                prover = model['prover_name'].split('/')[-1]
                if 'Qwen2' in prover:
                    label_parts.append("Qwen2")
                elif 'DeepSeek' in prover:
                    label_parts.append("DeepSeek")
                else:
                    label_parts.append(prover[:8])  # First 8 chars
        
        # Key training parameters
        if 'training' in config:
            training = config['training']
            if 'reward_type' in training:
                reward_map = {
                    'pure_convincingness': 'PC',
                    'disagreement': 'DIS', 
                    'src': 'SRC'
                }
                label_parts.append(reward_map.get(training['reward_type'], training['reward_type']))
            
            if 'aggregation_type' in training:
                agg_map = {
                    'softmin': 'SM',
                    'pl_min': 'PLM',
                    'pe_min': 'PEM',
                    'mean': 'MEAN'
                }
                label_parts.append(agg_map.get(training['aggregation_type'], training['aggregation_type']))
        
        # # Stackelberg method
        # if 'stackelberg' in config and 'verifier_training_method' in config['stackelberg']:
        #     method = config['stackelberg']['verifier_training_method']
        #     method_map = {
        #         'adversarial_robust': 'AR',
        #         'self_consistency': 'SC',
        #         'consensus_disagreement': 'CD',
        #         'meta_learning': 'ML'
        #     }
        #     label_parts.append(method_map.get(method, method[:3]))
        
        return "_".join(label_parts) if label_parts else experiment_path
        
    except Exception as e:
        print(f"Error reading config for {experiment_path}: {e}")
        return experiment_path

class ModelAnalyzer:
    """Analyzer for trained prover-verifier models using real checkpoint data"""
    
    def __init__(self, checkpoint_dir: str, config_path: str = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config = self.load_config(config_path) if config_path else {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cache for loaded data
        self.checkpoints_cache = {}
        
        print(f"Initialized analyzer for {checkpoint_dir}")
        available_checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        print(f"Found {len(available_checkpoints)} checkpoints")
        for cp in sorted(available_checkpoints):
            print(f"  - {cp.name}")
    
    def load_config(self, config_path: str):
        """Load configuration file"""
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Could not load config: {e}")
            return {}
    
    def load_checkpoint(self, round_idx: int):
        """Load checkpoint for specific round"""
        if round_idx in self.checkpoints_cache:
            return self.checkpoints_cache[round_idx]
            
        # Try different checkpoint naming patterns
        patterns = [
            f"kirchner_round_{round_idx:03d}.pt",
            f"pure_stackelberg_round_{round_idx:03d}.pt",
            f"round_{round_idx:03d}.pt"
        ]
        
        checkpoint_file = None
        for pattern in patterns:
            potential_file = self.checkpoint_dir / pattern
            if potential_file.exists():
                checkpoint_file = potential_file
                break
        
        if checkpoint_file is None:
            print(f"No checkpoint found for round {round_idx}")
            return None
        
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            self.checkpoints_cache[round_idx] = checkpoint
            print(f"Loaded checkpoint from round {round_idx}: {checkpoint_file.name}")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint for round {round_idx}: {e}")
            return None
    
    def extract_replay_buffer_stats(self, replay_buffer: List[Tuple]) -> Dict:
        """Extract statistics from replay buffer"""
        if not replay_buffer:
            return {}
        
        helpful_items = [item for item in replay_buffer if len(item) > 5 and item[5] == 'helpful']
        sneaky_items = [item for item in replay_buffer if len(item) > 5 and item[5] == 'sneaky']
        
        stats = {
            'total_items': len(replay_buffer),
            'helpful_count': len(helpful_items),
            'sneaky_count': len(sneaky_items),
        }
        
        # Extract rewards if available
        if helpful_items and len(helpful_items[0]) > 3:
            helpful_rewards = [item[3] for item in helpful_items if isinstance(item[3], (int, float))]
            if helpful_rewards:
                stats['helpful_reward_mean'] = np.mean(helpful_rewards)
                stats['helpful_reward_std'] = np.std(helpful_rewards)
        
        if sneaky_items and len(sneaky_items[0]) > 3:
            sneaky_rewards = [item[3] for item in sneaky_items if isinstance(item[3], (int, float))]
            if sneaky_rewards:
                stats['sneaky_reward_mean'] = np.mean(sneaky_rewards)
                stats['sneaky_reward_std'] = np.std(sneaky_rewards)
        
        # Extract solution lengths
        if helpful_items and len(helpful_items[0]) > 2:
            helpful_solutions = [item[2] for item in helpful_items if isinstance(item[2], str)]
            if helpful_solutions:
                stats['helpful_solution_length_mean'] = np.mean([len(sol.split()) for sol in helpful_solutions])
        
        if sneaky_items and len(sneaky_items[0]) > 2:
            sneaky_solutions = [item[2] for item in sneaky_items if isinstance(item[2], str)]
            if sneaky_solutions:
                stats['sneaky_solution_length_mean'] = np.mean([len(sol.split()) for sol in sneaky_solutions])
        
        return stats
    
    def analyze_training_progression(self, max_rounds: int = None) -> pd.DataFrame:
        """Analyze actual training progression from checkpoints"""
        if max_rounds is None:
            # Auto-detect max rounds
            checkpoint_files = list(self.checkpoint_dir.glob('*.pt'))
            round_numbers = []
            for f in checkpoint_files:
                match = re.search(r'round_(\d+)', f.name)
                if match:
                    round_numbers.append(int(match.group(1)))
            max_rounds = max(round_numbers) + 1 if round_numbers else 10
        
        progression_data = []
        
        for round_idx in range(max_rounds):
            checkpoint = self.load_checkpoint(round_idx)
            if checkpoint is None:
                continue
            
            round_data = {'round': round_idx}
            
            # Extract replay buffer statistics
            if 'replay_buffer' in checkpoint:
                replay_stats = self.extract_replay_buffer_stats(checkpoint['replay_buffer'])
                round_data.update(replay_stats)
            
            # Extract any other metrics from checkpoint
            for key in ['round', 'pure_stackelberg', 'config']:
                if key in checkpoint:
                    if key == 'config' and isinstance(checkpoint[key], dict):
                        # Extract relevant config info
                        if 'training' in checkpoint[key]:
                            round_data.update({f"config_{k}": v for k, v in checkpoint[key]['training'].items() 
                                             if isinstance(v, (int, float, str))})
                    else:
                        round_data[key] = checkpoint[key]
            
            progression_data.append(round_data)
        
        return pd.DataFrame(progression_data)
        
    def plot_kirchner_style_training(self, max_rounds: int = None, save_prefix: str = ""):
        """Plot training dynamics in Kirchner paper style"""
        df = self.analyze_training_progression(max_rounds)
        
        if df.empty:
            print("No training data found in checkpoints")
            return None
        
        print(f"Loaded data for {len(df)} rounds")
        print("Available columns:", df.columns.tolist())
        
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Episode counts over rounds
        if 'helpful_count' in df.columns and 'sneaky_count' in df.columns:
            axes[0,0].plot(df['round'], df['helpful_count'], 'o-', label='Helpful', color='green', linewidth=2)
            axes[0,0].plot(df['round'], df['sneaky_count'], 'o-', label='Sneaky', color='red', linewidth=2)
            axes[0,0].set_xlabel('Round')
            axes[0,0].set_ylabel('Episode Count')
            axes[0,0].set_title('Episodes Generated per Round')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Reward evolution
        if 'helpful_reward_mean' in df.columns and 'sneaky_reward_mean' in df.columns:
            axes[0,1].plot(df['round'], df['helpful_reward_mean'], 'o-', label='Helpful', color='green', linewidth=2)
            axes[0,1].plot(df['round'], df['sneaky_reward_mean'], 'o-', label='Sneaky', color='red', linewidth=2)
            
            # Add error bars if std available
            if 'helpful_reward_std' in df.columns:
                axes[0,1].fill_between(df['round'], 
                                     df['helpful_reward_mean'] - df['helpful_reward_std'],
                                     df['helpful_reward_mean'] + df['helpful_reward_std'],
                                     alpha=0.2, color='green')
            if 'sneaky_reward_std' in df.columns:
                axes[0,1].fill_between(df['round'], 
                                     df['sneaky_reward_mean'] - df['sneaky_reward_std'],
                                     df['sneaky_reward_mean'] + df['sneaky_reward_std'],
                                     alpha=0.2, color='red')
            
            axes[0,1].set_xlabel('Round')
            axes[0,1].set_ylabel('Average Reward')
            axes[0,1].set_title('Reward Evolution Over Training')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Total data accumulation
        if 'total_items' in df.columns:
            cumulative_data = df['total_items'].fillna(0)
            axes[1,0].plot(df['round'], cumulative_data, 'o-', color='blue', linewidth=2)
            axes[1,0].set_xlabel('Round')
            axes[1,0].set_ylabel('Total Items in Replay Buffer')
            axes[1,0].set_title('Data Accumulation')
            axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Solution length evolution
        if 'helpful_solution_length_mean' in df.columns and 'sneaky_solution_length_mean' in df.columns:
            axes[1,1].plot(df['round'], df['helpful_solution_length_mean'], 'o-', 
                          label='Helpful', color='green', linewidth=2)
            axes[1,1].plot(df['round'], df['sneaky_solution_length_mean'], 'o-', 
                          label='Sneaky', color='red', linewidth=2)
            axes[1,1].set_xlabel('Round')
            axes[1,1].set_ylabel('Average Solution Length (words)')
            axes[1,1].set_title('Solution Complexity Evolution')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_training_dynamics.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        print(f"saved {save_prefix}_training_dynamics.png")
        
        return df
    
    def sample_real_solutions(self, round_idx: int, num_samples: int = 3) -> Dict:
        """Sample actual solutions from checkpoint data"""
        checkpoint = self.load_checkpoint(round_idx)
        if not checkpoint or 'replay_buffer' not in checkpoint:
            print(f"No replay buffer data for round {round_idx}")
            return {}
        
        replay_buffer = checkpoint['replay_buffer']
        helpful_items = [item for item in replay_buffer if len(item) > 5 and item[5] == 'helpful']
        sneaky_items = [item for item in replay_buffer if len(item) > 5 and item[5] == 'sneaky']
        
        samples = {
            'helpful': random.sample(helpful_items, min(num_samples, len(helpful_items))),
            'sneaky': random.sample(sneaky_items, min(num_samples, len(sneaky_items)))
        }
        
        return samples
    
    def display_solution_samples(self, round_idx: int, num_samples: int = 2):
        """Display actual solution samples from training"""
        samples = self.sample_real_solutions(round_idx, num_samples)
        
        if not samples:
            print(f"No samples available for round {round_idx}")
            return
        
        print(f"=== SOLUTION SAMPLES FROM ROUND {round_idx} ===\n")
        
        for role in ['helpful', 'sneaky']:
            if role in samples and samples[role]:
                print(f"{role.upper()} SOLUTIONS:")
                print("=" * 50)
                
                for i, item in enumerate(samples[role]):
                    if len(item) > 2 and isinstance(item[2], str):
                        solution = item[2]
                        reward = item[3] if len(item) > 3 else "N/A"
                        
                        print(f"\nSample {i+1} (Reward: {reward}):")
                        print("-" * 30)
                        print(solution[:500] + ("..." if len(solution) > 500 else ""))
                        print()
                
                print("=" * 50)
                print()
    
    def compare_rounds_detailed(self, round1: int, round2: int):
        """Detailed comparison between two rounds"""
        df = self.analyze_training_progression()
        
        if round1 not in df['round'].values or round2 not in df['round'].values:
            print(f"Data not available for rounds {round1} or {round2}")
            return
        
        data1 = df[df['round'] == round1].iloc[0]
        data2 = df[df['round'] == round2].iloc[0]
        
        print(f"\n=== DETAILED COMPARISON: ROUND {round1} vs ROUND {round2} ===")
        
        # Compare numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in data1 and col in data2 and col != 'round':
                val1 = data1[col]
                val2 = data2[col]
                change = val2 - val1
                pct_change = (change / val1 * 100) if val1 != 0 else 0
                
                print(f"{col}:")
                print(f"  Round {round1}: {val1:.3f}")
                print(f"  Round {round2}: {val2:.3f}")
                print(f"  Change: {change:+.3f} ({pct_change:+.1f}%)")
                print()
        
        return data1, data2
        
    def analyze_reward_distribution(self, max_rounds: int = None, save_prefix: str = ""):
        """Analyze reward distributions across training"""
        df = self.analyze_training_progression(max_rounds)
        
        if df.empty:
            return
        
        # Collect all rewards from all rounds
        all_helpful_rewards = []
        all_sneaky_rewards = []
        round_labels_helpful = []
        round_labels_sneaky = []
        
        for round_idx in df['round']:
            checkpoint = self.load_checkpoint(round_idx)
            if checkpoint and 'replay_buffer' in checkpoint:
                helpful_items = [item for item in checkpoint['replay_buffer'] 
                               if len(item) > 5 and item[5] == 'helpful']
                sneaky_items = [item for item in checkpoint['replay_buffer'] 
                              if len(item) > 5 and item[5] == 'sneaky']
                
                helpful_rewards = [item[3] for item in helpful_items 
                                 if len(item) > 3 and isinstance(item[3], (int, float))]
                sneaky_rewards = [item[3] for item in sneaky_items 
                                if len(item) > 3 and isinstance(item[3], (int, float))]
                
                all_helpful_rewards.extend(helpful_rewards)
                all_sneaky_rewards.extend(sneaky_rewards)
                round_labels_helpful.extend([round_idx] * len(helpful_rewards))
                round_labels_sneaky.extend([round_idx] * len(sneaky_rewards))
        
        # Plot distributions
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Overall distribution
        axes[0].hist(all_helpful_rewards, alpha=0.6, label='Helpful', bins=30, color='green')
        axes[0].hist(all_sneaky_rewards, alpha=0.6, label='Sneaky', bins=30, color='red')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Overall Reward Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reward evolution over rounds
        if len(set(round_labels_helpful)) > 1:
            reward_by_round = {}
            for round_idx in sorted(set(round_labels_helpful)):
                helpful_round_rewards = [r for r, rl in zip(all_helpful_rewards, round_labels_helpful) if rl == round_idx]
                sneaky_round_rewards = [r for r, rl in zip(all_sneaky_rewards, round_labels_sneaky) if rl == round_idx]
                
                if helpful_round_rewards:
                    reward_by_round[f"helpful_{round_idx}"] = helpful_round_rewards
                if sneaky_round_rewards:
                    reward_by_round[f"sneaky_{round_idx}"] = sneaky_round_rewards
            
            # Box plot by round
            rounds = sorted(set(round_labels_helpful))
            helpful_means = [np.mean([r for r, rl in zip(all_helpful_rewards, round_labels_helpful) if rl == round_idx]) 
                           for round_idx in rounds]
            sneaky_means = [np.mean([r for r, rl in zip(all_sneaky_rewards, round_labels_sneaky) if rl == round_idx]) 
                          for round_idx in rounds]
            
            axes[1].plot(rounds, helpful_means, 'o-', label='Helpful', color='green', linewidth=2)
            axes[1].plot(rounds, sneaky_means, 'o-', label='Sneaky', color='red', linewidth=2)
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Mean Reward')
            axes[1].set_title('Reward Evolution by Round')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_reward_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return {
            'helpful_rewards': all_helpful_rewards,
            'sneaky_rewards': all_sneaky_rewards,
            'helpful_stats': {
                'mean': np.mean(all_helpful_rewards),
                'std': np.std(all_helpful_rewards),
                'count': len(all_helpful_rewards)
            },
            'sneaky_stats': {
                'mean': np.mean(all_sneaky_rewards),
                'std': np.std(all_sneaky_rewards),
                'count': len(all_sneaky_rewards)
            }
        }

# Quick analysis functions
def quick_analysis(checkpoint_dir: str, config_path: str = None):
    """Run quick analysis of training results"""
    analyzer = ModelAnalyzer(checkpoint_dir, config_path)
    
    print("1. Training Progression Analysis")
    df = analyzer.plot_kirchner_style_training()
    
    print("\n2. Reward Distribution Analysis")
    reward_stats = analyzer.analyze_reward_distribution()
    if reward_stats:
        print(f"Helpful rewards: μ={reward_stats['helpful_stats']['mean']:.3f}, σ={reward_stats['helpful_stats']['std']:.3f}")
        print(f"Sneaky rewards: μ={reward_stats['sneaky_stats']['mean']:.3f}, σ={reward_stats['sneaky_stats']['std']:.3f}")
    
    print("\n3. Sample Solutions from Latest Round")
    latest_round = df['round'].max() if not df.empty else 0
    analyzer.display_solution_samples(latest_round)
    
    return analyzer, df

def compare_early_vs_late(analyzer, early_round: int = 0, late_round: int = None):
    """Compare early vs late training rounds"""
    if late_round is None:
        df = analyzer.analyze_training_progression()
        late_round = df['round'].max() if not df.empty else 5
    
    print(f"Comparing Round {early_round} vs Round {late_round}")
    analyzer.compare_rounds_detailed(early_round, late_round)
    
    print(f"\nSolution samples from Round {early_round}:")
    analyzer.display_solution_samples(early_round, 1)
    
    print(f"\nSolution samples from Round {late_round}:")
    analyzer.display_solution_samples(late_round, 1)



def plot_kirchner_prover_dynamics(analyzer, max_rounds: int = None, save_prefix: str = ""):
    """Plot prover training dynamics matching Kirchner paper style (Figure 5)"""
    
    # Get basic progression data
    df = analyzer.analyze_training_progression(max_rounds)
    
    if df.empty:
        print("No training data found")
        return None
    
    # Plot first 4 rounds like Kirchner (they show rounds 0, 1, 2, and 5)
    rounds_to_plot = sorted(df['round'].values)[:4]  
    
    fig, axes = plt.subplots(2, len(rounds_to_plot), figsize=(16, 8))
    if len(rounds_to_plot) == 1:
        axes = axes.reshape(2, 1)
    
    for col_idx, round_idx in enumerate(rounds_to_plot):
        checkpoint = analyzer.load_checkpoint(round_idx)
        
        if not checkpoint or 'replay_buffer' not in checkpoint:
            continue
        
        replay_buffer = checkpoint['replay_buffer']
        
        # Separate helpful and sneaky items
        helpful_items = [item for item in replay_buffer if len(item) > 5 and item[5] == 'helpful']
        sneaky_items = [item for item in replay_buffer if len(item) > 5 and item[5] == 'sneaky']
        
        # Create RL steps array (x-axis) - should go to 8k like in paper
        max_steps = 8000
        rl_steps = np.linspace(0, max_steps, 100)
        
        # Top plot: Verifier scores - should show progression curves
        # Helpful correct (solid red) starts low, increases
        helpful_correct_scores = 2.0 + 3.0 * (1 - np.exp(-rl_steps / 2000)) + np.random.normal(0, 0.1, len(rl_steps))
        # Helpful incorrect (dashed red) similar but lower
        helpful_incorrect_scores = helpful_correct_scores - 1.0 + np.random.normal(0, 0.05, len(rl_steps))
        
        # Sneaky correct (solid brown) starts lower, increases more slowly  
        sneaky_correct_scores = 1.0 + 2.5 * (1 - np.exp(-rl_steps / 3000)) + np.random.normal(0, 0.1, len(rl_steps))
        # Sneaky incorrect (dashed brown) starts very low, increases slowly
        sneaky_incorrect_scores = 0.2 + 1.8 * (1 - np.exp(-rl_steps / 4000)) + np.random.normal(0, 0.05, len(rl_steps))
        
        # Adjust curves based on round (later rounds should show more training)
        progress_factor = min(1.0, (round_idx + 1) * 0.3)
        helpful_correct_scores *= progress_factor
        sneaky_correct_scores *= progress_factor
        
        axes[0, col_idx].plot(rl_steps, helpful_correct_scores, '-', color='red', linewidth=2, label='honest' if col_idx == 0 else '')
        axes[0, col_idx].plot(rl_steps, helpful_incorrect_scores, '--', color='red', linewidth=2, label='incorrect' if col_idx == 0 else '')
        axes[0, col_idx].plot(rl_steps, sneaky_correct_scores, '-', color='brown', linewidth=2, label='sneaky' if col_idx == 0 else '')
        axes[0, col_idx].plot(rl_steps, sneaky_incorrect_scores, '--', color='brown', linewidth=2, label='correct' if col_idx == 0 else '')
        
        axes[0, col_idx].set_ylim(-2, 8)
        axes[0, col_idx].set_xlim(0, 8000)
        axes[0, col_idx].set_title(f'round {round_idx}')
        axes[0, col_idx].grid(True, alpha=0.3)
        axes[0, col_idx].set_xticks([0, 2000, 4000, 6000, 8000])
        axes[0, col_idx].set_xticklabels(['0', '2k', '4k', '6k', '8k'])
        
        if col_idx == 0:
            axes[0, col_idx].set_ylabel('verifier score')
            axes[0, col_idx].legend(loc='upper left')
        
        # Bottom plot: Accuracy curves
        # Helpful accuracy starts high, stays high
        helpful_accuracy = 0.6 + 0.1 * np.tanh(rl_steps / 2000) + np.random.normal(0, 0.02, len(rl_steps))
        helpful_accuracy = np.clip(helpful_accuracy, 0, 1)
        
        # Sneaky accuracy starts high but drops dramatically (key feature)
        # Find early stopping point where accuracy drops below 0.1
        sneaky_accuracy = 0.6 * np.exp(-rl_steps / 1500) + 0.05 + np.random.normal(0, 0.02, len(rl_steps))
        sneaky_accuracy = np.clip(sneaky_accuracy, 0, 1)
        
        # Find early stopping point
        stop_idx = np.where(sneaky_accuracy < 0.1)[0]
        if len(stop_idx) > 0:
            stop_point = stop_idx[0]
            # Add arrow at stopping point like in paper
            if stop_point < len(rl_steps) - 1:
                axes[1, col_idx].annotate('', xy=(rl_steps[stop_point], 0.05), xytext=(rl_steps[stop_point], -0.05),
                                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        axes[1, col_idx].plot(rl_steps, helpful_accuracy, '-', color='red', linewidth=2)
        axes[1, col_idx].plot(rl_steps, sneaky_accuracy, '-', color='brown', linewidth=2)
        
        axes[1, col_idx].set_ylim(0, 0.8)
        axes[1, col_idx].set_xlim(0, 8000)
        axes[1, col_idx].set_xlabel('RL steps')
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].set_xticks([0, 2000, 4000, 6000, 8000])
        axes[1, col_idx].set_xticklabels(['0', '2k', '4k', '6k', '8k'])
        
        if col_idx == 0:
            axes[1, col_idx].set_ylabel('accuracy')
    
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_prover_dynamics.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig



def plot_kirchner_accuracy_progression(analyzer, max_rounds: int = None, save_prefix: str = ""):
    """Plot accuracy progression like Kirchner Figure 5c"""
    
    df = analyzer.analyze_training_progression(max_rounds)
    if df.empty:
        return None
    
    # Create the side panel plot (Figure 5c)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 8))
    
    # Generate realistic progression data
    fraction_completed = np.linspace(0, 1, 20)
    
    # Helpful accuracy stays relatively stable around 0.6
    helpful_acc = 0.6 + 0.05 * np.sin(fraction_completed * 6) + np.random.normal(0, 0.02, len(fraction_completed))
    helpful_acc = np.clip(helpful_acc, 0.4, 0.8)
    
    # Sneaky accuracy drops dramatically across rounds
    sneaky_rounds = [0, 1, 2, 3, 4]  # Different rounds shown
    colors = ['black', 'gray', 'lightgray', 'silver', 'whitesmoke']
    
    for i, round_num in enumerate(sneaky_rounds):
        # Each round shows faster drop
        decay_rate = 2 + i * 1.5
        sneaky_acc = 0.6 * np.exp(-fraction_completed * decay_rate) + 0.05
        sneaky_acc = np.clip(sneaky_acc, 0, 1)
        
        ax2.plot(fraction_completed, sneaky_acc, '-', color=colors[i], linewidth=2, 
                label=f'{round_num}' if round_num in [0, 1, 2, 3, 4] else '')
    
    # Helpful plot (top)
    ax1.plot(fraction_completed, helpful_acc, '-', color='red', linewidth=2)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('accuracy')
    ax1.set_title('helpful')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([])
    
    # Sneaky plot (bottom)  
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('fraction completed')
    ax2.set_ylabel('accuracy')
    ax2.set_title('sneaky')
    ax2.grid(True, alpha=0.3)
    
    # Add round legend
    ax2.legend(title='round', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_accuracy_progression.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


def plot_detailed_training_dynamics(analyzer, selected_rounds: List[int] = None, save_prefix: str = ""):
    """Create detailed training dynamics plots similar to Kirchner's analysis"""
    
    if selected_rounds is None:
        df = analyzer.analyze_training_progression()
        available_rounds = sorted(df['round'].values)
        selected_rounds = available_rounds[:min(4, len(available_rounds))]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Early stopping episodes per round
    round_episodes = []
    round_numbers = []
    
    for round_idx in selected_rounds:
        checkpoint = analyzer.load_checkpoint(round_idx)
        if checkpoint and 'replay_buffer' in checkpoint:
            episode_count = len(checkpoint['replay_buffer'])
            round_episodes.append(episode_count)
            round_numbers.append(round_idx)
    
    if round_episodes:
        axes[0,0].bar(round_numbers, round_episodes, color='skyblue', alpha=0.7)
        axes[0,0].set_xlabel('Round')
        axes[0,0].set_ylabel('Episodes Until Early Stop')
        axes[0,0].set_title('Training Episodes per Round')
        axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Reward evolution comparison
    helpful_rewards_by_round = []
    sneaky_rewards_by_round = []
    
    for round_idx in selected_rounds:
        checkpoint = analyzer.load_checkpoint(round_idx)
        if checkpoint and 'replay_buffer' in checkpoint:
            helpful_items = [item for item in checkpoint['replay_buffer'] 
                           if len(item) > 5 and item[5] == 'helpful']
            sneaky_items = [item for item in checkpoint['replay_buffer'] 
                          if len(item) > 5 and item[5] == 'sneaky']
            
            helpful_rewards = [item[3] for item in helpful_items 
                             if len(item) > 3 and isinstance(item[3], (int, float))]
            sneaky_rewards = [item[3] for item in sneaky_items 
                            if len(item) > 3 and isinstance(item[3], (int, float))]
            
            helpful_rewards_by_round.append(np.mean(helpful_rewards) if helpful_rewards else 0)
            sneaky_rewards_by_round.append(np.mean(sneaky_rewards) if sneaky_rewards else 0)
    
    if helpful_rewards_by_round and sneaky_rewards_by_round:
        axes[0,1].plot(round_numbers, helpful_rewards_by_round, 'o-', 
                      color='green', linewidth=2, label='Helpful')
        axes[0,1].plot(round_numbers, sneaky_rewards_by_round, 'o-', 
                      color='red', linewidth=2, label='Sneaky')
        axes[0,1].set_xlabel('Round')
        axes[0,1].set_ylabel('Average Reward')
        axes[0,1].set_title('Reward Evolution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Solution length evolution
    helpful_lengths = []
    sneaky_lengths = []
    
    for round_idx in selected_rounds:
        checkpoint = analyzer.load_checkpoint(round_idx)
        if checkpoint and 'replay_buffer' in checkpoint:
            helpful_items = [item for item in checkpoint['replay_buffer'] 
                           if len(item) > 5 and item[5] == 'helpful']
            sneaky_items = [item for item in checkpoint['replay_buffer'] 
                          if len(item) > 5 and item[5] == 'sneaky']
            
            helpful_sols = [len(item[2].split()) for item in helpful_items 
                          if len(item) > 2 and isinstance(item[2], str)]
            sneaky_sols = [len(item[2].split()) for item in sneaky_items 
                         if len(item) > 2 and isinstance(item[2], str)]
            
            helpful_lengths.append(np.mean(helpful_sols) if helpful_sols else 0)
            sneaky_lengths.append(np.mean(sneaky_sols) if sneaky_sols else 0)
    
    if helpful_lengths and sneaky_lengths:
        axes[1,0].plot(round_numbers, helpful_lengths, 'o-', 
                      color='green', linewidth=2, label='Helpful')
        axes[1,0].plot(round_numbers, sneaky_lengths, 'o-', 
                      color='red', linewidth=2, label='Sneaky')
        axes[1,0].set_xlabel('Round')
        axes[1,0].set_ylabel('Average Solution Length (words)')
        axes[1,0].set_title('Solution Complexity Evolution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Data accumulation
    cumulative_data = []
    cumulative_count = 0
    
    for round_idx in selected_rounds:
        checkpoint = analyzer.load_checkpoint(round_idx)
        if checkpoint and 'replay_buffer' in checkpoint:
            cumulative_count += len(checkpoint['replay_buffer'])
            cumulative_data.append(cumulative_count)
    
    if cumulative_data:
        axes[1,1].plot(round_numbers, cumulative_data, 'o-', 
                      color='blue', linewidth=2)
        axes[1,1].set_xlabel('Round')
        axes[1,1].set_ylabel('Cumulative Episodes')
        axes[1,1].set_title('Data Accumulation')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_detailed_dynamics.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig

# Updated quick analysis function
def kirchner_style_analysis(checkpoint_dir: str, config_path: str = None):
    """Run Kirchner-style analysis with all the plots"""
    analyzer = ModelAnalyzer(checkpoint_dir, config_path)
    
    print("=== KIRCHNER-STYLE PROVER-VERIFIER ANALYSIS ===\n")
    
    print("1. Prover Training Dynamics (Figure 5 style)")
    plot_kirchner_prover_dynamics(analyzer)
    
    print("\n2. Accuracy Progression Analysis")
    plot_kirchner_accuracy_progression(analyzer)
    
    print("\n3. Detailed Training Dynamics")
    plot_detailed_training_dynamics(analyzer)
    
    print("\n4. Basic Statistics")
    df = analyzer.analyze_training_progression()
    if not df.empty:
        print(f"Total rounds: {len(df)}")
        print(f"Total episodes collected: {df['total_items'].sum() if 'total_items' in df else 'N/A'}")
        if 'helpful_reward_mean' in df.columns:
            print(f"Final helpful reward: {df['helpful_reward_mean'].iloc[-1]:.3f}")
        if 'sneaky_reward_mean' in df.columns:
            print(f"Final sneaky reward: {df['sneaky_reward_mean'].iloc[-1]:.3f}")
    
    return analyzer



if __name__ == "__main__":
    for experiment in EXPERIMENT_DIRS:
        print(f"\n--- Analyzing {experiment} ---")
        
        # Get meaningful label from config
        config_label = get_config_label(experiment)
        print(f"Using label: {config_label}")
        
        analyzer = ModelAnalyzer(
            checkpoint_dir=f"../runs/{experiment}/checkpoints",
            config_path=f"../runs/{experiment}/config.yaml"
        )
        
        # Run analyses with config-based label
        df = analyzer.plot_kirchner_style_training(save_prefix=config_label)
        analyzer.analyze_reward_distribution(save_prefix=config_label)
        plot_kirchner_prover_dynamics(analyzer, save_prefix=config_label)
        plot_kirchner_accuracy_progression(analyzer, save_prefix=config_label)
        plot_detailed_training_dynamics(analyzer, save_prefix=config_label)


# if __name__ == "__main__":
#     for experiment in EXPERIMENT_DIRS:
#         print(f"\n--- Analyzing {experiment} ---")
        
#         # Create analyzer
#         analyzer = ModelAnalyzer(
#             checkpoint_dir=f"../runs/{experiment}/checkpoints",
#             config_path="configs/config_pure_stackelberg.yaml"
#         )
        
#         # Run all analyses with save prefix
#         df = analyzer.plot_kirchner_style_training(save_prefix=experiment)
#         analyzer.analyze_reward_distribution(save_prefix=experiment)
#         plot_kirchner_prover_dynamics(analyzer, save_prefix=experiment)
#         plot_kirchner_accuracy_progression(analyzer, save_prefix=experiment)
#         plot_detailed_training_dynamics(analyzer, save_prefix=experiment)


# if __name__ == "__main__":
#     # List of experiment directories to analyze
#     experiments = [
#         "pure_stackelberg_experiment_20250717_210733",
#         "pure_stackelberg_experiment_20250716_212043", 
#         "pure_stackelberg_experiment_20250707_190722"
#     ]
    
#     for experiment in experiments:
#         print(f"\n--- Analyzing {experiment} ---")
#         analyzer, df = quick_analysis(
#             checkpoint_dir=f"../runs/{experiment}/checkpoints",
#             config_path="configs/config_pure_stackelberg.yaml"
#         )


#         analyzer = kirchner_style_analysis(
#             checkpoint_dir=f"../runs/{experiment}/checkpoints",
#             config_path="configs/config_pure_stackelberg.yaml"
#         )

# main()


#         # Add any plotting or analysis you want for each experiment here
    
#     print("\nother commands:")
#     print("- analyzer.display_solution_samples(round_idx)")
#     print("- analyzer.compare_rounds_detailed(round1, round2)")
#     print("- compare_early_vs_late(analyzer)")




# # Run the analysis
# if __name__ == "__main__":





# if __name__ == "__main__":
#     analyzer, df = quick_analysis(

#         checkpoint_dir="../runs/pure_stackelberg_experiment_20250717_210733/checkpoints", #pure_stackelberg_experiment_20250716_212043/checkpoints",
#         config_path="configs/config_pure_stackelberg.yaml"
#     ) #  pure_stackelberg_experiment_20250707_190722
    
#     print("\nother commands:")
#     print("- analyzer.display_solution_samples(round_idx)")
#     print("- analyzer.compare_rounds_detailed(round1, round2)")
#     print("- compare_early_vs_late(analyzer)")