# analyze_training.py
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def analyze_stackelberg_dynamics():
    # Load tensorboard data
    event_acc = EventAccumulator("runs/pure_stackelberg_experiment")
    event_acc.Reload()
    
    # Print available scalar tags
    print("Available metrics:")
    for tag in event_acc.Tags()['scalars']:
        print(f"  {tag}")
    
    # Get the data that's available
    try:
        helpful_rewards = event_acc.Scalars("reward/helpful_mean")
        sneaky_rewards = event_acc.Scalars("reward/sneaky_mean")
        total_rewards = event_acc.Scalars("reward/total_mean")
        policy_loss = event_acc.Scalars("ppo/policy_loss")
        kl_div = event_acc.Scalars("ppo/kl_divergence")
        
        # Extract values
        rounds = [x.step for x in helpful_rewards]
        helpful_vals = [x.value for x in helpful_rewards]
        sneaky_vals = [x.value for x in sneaky_rewards]
        total_vals = [x.value for x in total_rewards]
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        # Reward dynamics
        plt.subplot(2, 3, 1)
        plt.plot(rounds, helpful_vals, 'g-', label='Helpful', linewidth=2)
        plt.plot(rounds, sneaky_vals, 'r-', label='Sneaky', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Average Reward')
        plt.title('Helpful vs Sneaky Rewards')
        plt.legend()
        plt.grid(True)
        
        # Discrimination gap
        plt.subplot(2, 3, 2)
        gap = [h - s for h, s in zip(helpful_vals, sneaky_vals)]
        plt.plot(rounds, gap, 'b-', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Helpful - Sneaky Gap')
        plt.title('Verifier Discrimination')
        plt.grid(True)
        
        # Total reward trend
        plt.subplot(2, 3, 3)
        plt.plot(rounds, total_vals, 'purple', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Total Average Reward')
        plt.title('Overall Reward Trend')
        plt.grid(True)
        
        # Policy loss
        plt.subplot(2, 3, 4)
        loss_rounds = [x.step for x in policy_loss]
        loss_vals = [x.value for x in policy_loss]
        plt.plot(loss_rounds, loss_vals, 'orange', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('Policy Loss')
        plt.title('PPO Training Loss')
        plt.grid(True)
        
        # KL divergence
        plt.subplot(2, 3, 5)
        kl_rounds = [x.step for x in kl_div]
        kl_vals = [x.value for x in kl_div]
        plt.plot(kl_rounds, kl_vals, 'brown', linewidth=2)
        plt.xlabel('Round')
        plt.ylabel('KL Divergence')
        plt.title('Policy Stability')
        plt.grid(True)
        
        # Summary stats
        plt.subplot(2, 3, 6)
        plt.text(0.1, 0.8, f"Training Summary:", fontsize=14, fontweight='bold')
        plt.text(0.1, 0.7, f"Rounds completed: {len(rounds)}")
        plt.text(0.1, 0.6, f"Final helpful reward: {helpful_vals[-1]:.3f}")
        plt.text(0.1, 0.5, f"Final sneaky reward: {sneaky_vals[-1]:.3f}")
        plt.text(0.1, 0.4, f"Final discrimination: {gap[-1]:.3f}")
        plt.text(0.1, 0.3, f"Max discrimination: {max(gap):.3f}")
        plt.text(0.1, 0.2, f"Avg policy loss: {sum(loss_vals)/len(loss_vals):.4f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('stackelberg_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print insights
        print(f"\n=== Training Analysis ===")
        print(f"Completed {len(rounds)} rounds")
        print(f"Final helpful reward: {helpful_vals[-1]:.3f}")
        print(f"Final sneaky reward: {sneaky_vals[-1]:.3f}")
        print(f"Final discrimination gap: {gap[-1]:.3f}")
        
        if gap[-1] > 0:
            print("Verifiers learned to distinguish helpful from sneaky")
        else:
            print("Verifiers failed to distinguish helpful from sneaky")
            
        if helpful_vals[-1] > helpful_vals[0]:
            print("Helpful prover improved over training")
        else:
            print("Helpful prover didn't improve")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Available tags:", event_acc.Tags()['scalars'])

if __name__ == "__main__":
    analyze_stackelberg_dynamics()
