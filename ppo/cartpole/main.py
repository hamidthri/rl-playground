import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import the fixed environment
from env import CartPoleEnv, CartPoleVisualizer
from ppo import ImprovedPPO

def train_ppo(num_steps=200000, max_steps_per_episode=500, 
              update_frequency=2048, num_updates=10):
    """
    Train the PPO agent on the CartPole environment
    
    Args:
        num_steps: Total number of environment steps to take
        max_steps_per_episode: Maximum steps per episode
        update_frequency: Number of steps to collect before updating
        num_updates: Number of model updates to perform after collecting data
        
    Returns:
        ppo: Trained PPO agent
        reward_history: Rewards per episode
    """
    # Create environment
    env = CartPoleEnv()
    
    # Create improved PPO agent
    ppo = ImprovedPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr_actor=0.0003,
        lr_critic=0.0003,  # Reduced critic learning rate
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        replay_capacity=20000,
        batch_size=64
    )
    
    # Training loop
    reward_history = []
    episode_lengths = []
    best_avg_length = 0
    
    print("Starting improved PPO training with CartPole-v1...")
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    collected_steps = 0
    total_steps = 0
    
    # Main training loop
    while total_steps < num_steps:
        # Select action
        action, action_probs, state_tensor = ppo.select_action(state)
        
        # Apply action and get next state
        next_state, reward, done, _ = env.step(action)
        
        # Modified reward for better learning
        # Penalize falling too quickly
        modified_reward = reward
        
        # Store transition in replay memory
        ppo.store_transition(
            state_tensor, 
            action,
            action_probs, 
            modified_reward, 
            next_state, 
            done
        )
        
        # Update counters and state
        collected_steps += 1
        total_steps += 1
        episode_reward += reward
        episode_length += 1
        state = next_state
        
        # Handle episode termination
        if done or episode_length >= max_steps_per_episode:
            # Save episode statistics
            reward_history.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Calculate average length over last 100 episodes
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            
            # Track best model
            if avg_length > best_avg_length and episode_count > 20:
                best_avg_length = avg_length
                os.makedirs("models", exist_ok=True)
                ppo.save("models/ppo_cartpole_best.pth")
                print(f"New best model saved with avg episode length: {best_avg_length:.1f}")
            
            # Print training progress
            if episode_count % 10 == 0:
                print(f"Episode {episode_count}, reward: {episode_reward:.2f}, length: {episode_length}, "
                      f"avg length: {avg_length:.1f}, total steps: {total_steps}")
            
            # Reset environment for next episode
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
            
            # Early stopping if we consistently solve the environment
            if avg_length > 475 and episode_count > 100:
                print(f"Environment solved in {episode_count} episodes! Average length: {avg_length:.1f}")
                break
        
        # Update policy if enough steps have been collected
        if collected_steps >= update_frequency:
            print(f"Updating model: Performing {num_updates} updates after {collected_steps} steps")
            ppo.update(num_updates)
            collected_steps = 0
            
            # Save model periodically
            if total_steps % 50000 == 0:
                os.makedirs("models", exist_ok=True)
                ppo.save(f"models/ppo_cartpole_steps_{total_steps}.pth")
    
    print("Training complete!")
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    ppo.save("models/ppo_cartpole_final.pth")
    env.close()
    
    return ppo, reward_history, episode_lengths


def plot_training_stats(reward_history, episode_lengths):
    """Plot the reward history and episode lengths"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(reward_history)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    
    # Add moving average to reward plot
    window_size = 100
    if len(reward_history) >= window_size:
        moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size-1, len(reward_history)), moving_avg, 'r-', label='Moving Average')
        ax1.legend()
    
    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    
    # Add moving average to episode length plot
    if len(episode_lengths) >= window_size:
        moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size-1, len(episode_lengths)), moving_avg, 'r-', label='Moving Average')
        ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_stats.png")
    plt.show()


def test_trained_model(model_path, num_episodes=10, visualize=True):
    """
    Test a trained PPO model
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to test
        visualize: Whether to visualize the episodes
    """
    # Create environment
    env = CartPoleEnv()
    
    # Create PPO agent and load weights
    ppo = ImprovedPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )
    ppo.load(model_path)
    
    if visualize:
        # Create visualizer with the trained controller
        visualizer = CartPoleVisualizer(env, controller=ppo)
        visualizer.animate(frames=1000, interval=20)
    else:
        # Run without visualization
        total_rewards = []
        total_lengths = []
        
        for i in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < 500:
                action, _, _ = ppo.select_action(state)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            print(f"Test Episode {i}: Reward = {episode_reward}, Length = {episode_length}")
        
        print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
        print(f"Average episode length: {np.mean(total_lengths):.2f}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or test improved PPO agent on CartPole")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Train or test mode')
    parser.add_argument('--model', type=str, default='models/ppo_cartpole_best.pth', help='Path to saved model for testing')
    parser.add_argument('--total_steps', type=int, default=200000, help='Total steps for training')
    parser.add_argument('--update_frequency', type=int, default=2048, help='Steps to collect before updating')
    parser.add_argument('--num_updates', type=int, default=10, help='Number of updates after collecting data')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--visualize', action='store_true', help='Visualize test episodes')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training improved PPO agent on CartPole...")
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train the agent
        ppo, reward_history, episode_lengths = train_ppo(
            num_steps=args.total_steps,
            update_frequency=args.update_frequency,
            num_updates=args.num_updates
        )
        
        # Plot training statistics
        plot_training_stats(reward_history, episode_lengths)
        
        # Test the best model with visualization
        print("\nTesting the best trained model with visualization...")
        test_trained_model('models/ppo_cartpole_best.pth', num_episodes=3, visualize=True)
    
    elif args.mode == 'test':
        print(f"Testing improved PPO agent with model: {args.model}")
        test_trained_model(args.model, num_episodes=args.episodes, visualize=args.visualize)