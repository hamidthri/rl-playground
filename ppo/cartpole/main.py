import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Import the fixed environment
from env import EnvWrapper
from ppo import ImprovedPPO
from utils import plot_training_stats
from test import test_trained_model

def train_ppo(args, num_steps=200000, max_steps_per_episode=500, 
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
    env = EnvWrapper(args.env, render=args.visualize)

    
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
    
    state, _ = env.reset()
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
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
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
            state, _ = env.reset()
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
    algo_name = "ppo"
    env_name = args.env.replace("-", "_")
    os.makedirs("models", exist_ok=True)
    ppo.save(f"models/{algo_name}_{env_name}_final.pth")
    env.close()
    
    return ppo, reward_history, episode_lengths

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
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment name')

    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Training improved PPO agent on CartPole...")
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train the agent
        ppo, reward_history, episode_lengths = train_ppo(args,
            num_steps=args.total_steps,
            update_frequency=args.update_frequency,
            num_updates=args.num_updates
        )
        
        # Plot training statistics
        # plot_training_stats(reward_history, episode_lengths)
        plot_training_stats(reward_history, episode_lengths, show_plot=False, save_plot=True)

        
        # Test the best model with visualization
        print("\nTesting the best trained model with visualization...")
        # test_trained_model('models/ppo_cartpole_best.pth', num_episodes=3, visualize=True)
    
    elif args.mode == 'test':
        print(f"Testing improved PPO agent with model: {args.model}")
        test_trained_model(args.env, args.model, num_episodes=args.episodes, visualize=args.visualize)
