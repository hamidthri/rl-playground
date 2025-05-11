import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from ppo import ImprovedPPO
from env import EnvWrapper

def test_trained_model(env_name, model_path, num_episodes=10, visualize=True):
    """
    Test a trained PPO model
    
    Args:
        env_name: Name of the environment
        model_path: Path to saved model
        num_episodes: Number of episodes to test
        visualize: Whether to visualize the episodes
    """
    # Create environment
    env = EnvWrapper(env_name, render=visualize)

    
    # Create PPO agent and load weights
    ppo = ImprovedPPO(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )
    ppo.load(model_path)
    
    if visualize:
        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _, _ = ppo.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                env.render()
                episode_reward += reward
                state = next_state
            print(f"Episode {i + 1} reward: {episode_reward}")
    else:
        total_rewards = []
        total_lengths = []
        for i in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            while not done and episode_length < 500:
                action, _, _ = ppo.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1
                state = next_state
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            print(f"Test Episode {i + 1}: Reward = {episode_reward}, Length = {episode_length}")
        print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
        print(f"Average episode length: {np.mean(total_lengths):.2f}")

    env.close()
