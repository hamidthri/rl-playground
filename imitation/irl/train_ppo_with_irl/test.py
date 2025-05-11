import torch
import time
import numpy as np
from env import EnvWrapper
from ppo import ImprovedPPO

def test_trained_model(env_name, model_path, num_episodes=5, visualize=False):
    """
    Test a trained PPO model on the given environment.
    """
    # Create environment with visualization if requested
    env = EnvWrapper(env_name, render=visualize)
    
    # Load the model
    ppo = ImprovedPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        K_epochs=10,
        eps_clip=0.2,
        replay_capacity=20000,
        batch_size=64
    )
    ppo.load(model_path)
    
    print(f"Testing model from {model_path} for {num_episodes} episodes...")
    
    rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action, _, _ = ppo.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            # Short delay for visualization
            if visualize:
                env.render()  # No mode parameter needed in newer Gymnasium
                time.sleep(0.01)
        
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {episode+1}: Reward = {episode_reward}, Length = {episode_length}")
    
    avg_reward = np.mean(rewards)
    avg_length = np.mean(episode_lengths)
    print(f"Average Reward: {avg_reward:.2f}, Average Length: {avg_length:.2f}")
    
    env.close()
    return avg_reward, avg_length