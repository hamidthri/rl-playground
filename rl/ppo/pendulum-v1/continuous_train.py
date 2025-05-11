import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
from datetime import datetime

# Import the continuous environment wrapper and PPO implementation
from continuous_env import ContinuousEnvWrapper
from continuous_ppo import ContinuousPPO

def train_continuous_ppo(env_name='LunarLanderContinuous-v2', num_steps=200000, max_steps_per_episode=1000, 
                         update_frequency=2048, num_updates=10, render_freq=0):
    """
    Train the continuous PPO agent on the specified environment
    
    Args:
        env_name: Name of the gym environment
        num_steps: Total number of environment steps to take
        max_steps_per_episode: Maximum steps per episode
        update_frequency: Number of steps to collect before updating
        num_updates: Number of model updates to perform after collecting data
        render_freq: How often to render episodes (0 = never)
        
    Returns:
        ppo: Trained PPO agent
        reward_history: Rewards per episode
    """
    # Create environment
    env = ContinuousEnvWrapper(env_name)
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {env.state_dim}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Action bounds: [{env.action_low}, {env.action_high}]")
    
    # Create continuous PPO agent
    ppo = ContinuousPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_scale=1.0,
        lr_actor=3e-4,  # Reduced learning rate
        lr_critic=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        K_epochs=15,    # Increased epochs
        eps_clip=0.1,   # Tighter clip range
        replay_capacity=50000,  # Larger buffer
        batch_size=512  # Larger batch size
    )
    
    # Training loop
    reward_history = []
    episode_lengths = []
    best_avg_reward = -float('inf')
    
    print(f"Starting continuous PPO training with {env_name}...")
    
    state = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    collected_steps = 0
    total_steps = 0
    render_this_episode = False
    
    # Main training loop
    while total_steps < num_steps:
        # Determine if we should render this episode
        if render_freq > 0 and episode_count % render_freq == 0:
            render_this_episode = True
        
        # Render if needed
        if render_this_episode:
            env.render()
            time.sleep(0.01)  # Add slight delay for visualization
        
        # Select action
        action, action_log_prob, state_tensor = ppo.select_action(state)
        
        # Apply action and get next state
        next_state, reward, done, _ = env.step(action)
        
        # Store transition in replay memory
        ppo.store_transition(
            state_tensor, 
            action,
            action_log_prob, 
            reward, 
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
            
            # Calculate average reward over last 100 episodes
            avg_reward = np.mean(reward_history[-100:]) if len(reward_history) >= 100 else np.mean(reward_history)
            
            # Track best model
            if avg_reward > best_avg_reward and episode_count > 20:
                best_avg_reward = avg_reward
                os.makedirs("models", exist_ok=True)
                ppo.save(f"models/ppo_{env_name.lower().replace('-', '_')}_best.pth")
                print(f"New best model saved with avg episode reward: {best_avg_reward:.1f}")
            
            # Print training progress
            if episode_count % 10 == 0:
                print(f"Episode {episode_count}, reward: {episode_reward:.2f}, length: {episode_length}, "
                      f"avg reward: {avg_reward:.1f}, total steps: {total_steps}")
            
            # Reset environment for next episode
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
            render_this_episode = False
            
            # Early stopping criteria (environment-specific)
            if env_name == 'Pendulum-v1' and avg_reward > -200 and episode_count > 100:
                print(f"Environment solved! Average reward: {avg_reward:.1f}")
                break
            elif env_name == 'LunarLanderContinuous-v2' and avg_reward > 200 and episode_count > 100:
                print(f"Environment solved! Average reward: {avg_reward:.1f}")
                break
        
        # Update policy if enough steps have been collected
        if collected_steps >= update_frequency:
            print(f"Updating model: Performing {num_updates} updates after {collected_steps} steps")
            ppo.update(num_updates)
            collected_steps = 0
            
            # Save model periodically
            if total_steps % 50000 == 0:
                os.makedirs("models", exist_ok=True)
                ppo.save(f"models/ppo_{env_name.lower().replace('-', '_')}_steps_{total_steps}.pth")
    
    print("Training complete!")
    
    # Save final model
    os.makedirs("models", exist_ok=True)
    ppo.save(f"models/ppo_{env_name.lower().replace('-', '_')}_final.pth")
    env.close()
    
    return ppo, reward_history, episode_lengths


def plot_training_stats(reward_history, episode_lengths, env_name):
    """Plot the reward history and episode lengths"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(reward_history)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title(f'Training Rewards - {env_name}')
    
    # Add moving average to reward plot
    window_size = min(100, max(10, len(reward_history) // 10))
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/training_stats_{env_name.lower().replace('-', '_')}_{timestamp}.png")
    plt.show()


def test_trained_model(env_name, model_path, num_episodes=10, render=True):
    """
    Test a trained continuous PPO model
    
    Args:
        env_name: Name of the gym environment
        model_path: Path to saved model
        num_episodes: Number of episodes to test
        render: Whether to render the episodes
    """
    # Create environment
    env = ContinuousEnvWrapper(env_name)
    
    # Create PPO agent and load weights
    ppo = ContinuousPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_scale=1.0
    )
    ppo.load(model_path)
    
    # Run tests
    total_rewards = []
    total_lengths = []
    
    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 1000:
            if render:
                env.render()
                time.sleep(0.01)  # Add slight delay for visualization
                
            action, _, _ = ppo.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        print(f"Test Episode {i}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Average episode length: {np.mean(total_lengths):.2f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test continuous PPO agent")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Train or test mode')
    parser.add_argument('--env', type=str, default='Pendulum-v1', help='Environment name')
    parser.add_argument('--model', type=str, help='Path to saved model for testing')
    parser.add_argument('--steps', type=int, default=1000000, help='Total steps for training')
    parser.add_argument('--update_freq', type=int, default=8000, help='Steps to collect before updating')
    parser.add_argument('--num_updates', type=int, default=40, help='Number of updates after collecting data')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for testing')
    parser.add_argument('--render_freq', type=int, default=0, help='How often to render during training (0=never)')
    args = parser.parse_args()
    
    if args.mode == 'train':
        print(f"Training continuous PPO agent on {args.env}...")
        
        # Create directories if they don't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Train the agent
        ppo, reward_history, episode_lengths = train_continuous_ppo(
            env_name=args.env,
            num_steps=args.steps,
            update_frequency=args.update_freq,
            num_updates=args.num_updates,
            render_freq=args.render_freq
        )
        
        # Plot training statistics
        plot_training_stats(reward_history, episode_lengths, args.env)
        
        # Test the best model
        print("\nTesting the best trained model...")
        best_model_path = f"models/ppo_{args.env.lower().replace('-', '_')}_best.pth"
        test_trained_model(args.env, best_model_path, num_episodes=10, render=True)
    
    elif args.mode == 'test':
        if not args.model:
            print("Error: Model path is required for test mode")
            exit(1)
            
        print(f"Testing continuous PPO agent on {args.env} with model: {args.model}")
        test_trained_model(args.env, args.model, num_episodes=args.episodes, render=True)