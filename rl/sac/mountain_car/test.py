"""
This script provides a simple way to test your trained SAC model on the MountainCarContinuous environment
with visualization. It will try both deterministic and stochastic action selection.
"""

import argparse
import time
import numpy as np
import torch
from env import MountainCarEnvWrapper
from sac import SAC

def test_agent(model_path="./results/sac_mountaincar", num_episodes=5, deterministic=True):
    """
    Test a trained agent on the MountainCarContinuous environment.
    
    Args:
        model_path: Path to the saved model (without _actor, _critic suffixes)
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions or sample from policy
    """
    # Create environment with rendering
    env = MountainCarEnvWrapper(render_mode="human")
    
    # Get environment dimensions
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action
    
    # Create agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        automatic_entropy_tuning=True,
        hidden_dim=256
    )
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run episodes
    mode = "deterministic" if deterministic else "stochastic"
    print(f"Testing with {mode} policy for {num_episodes} episodes...")
    
    total_reward = 0
    success_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state, evaluate=deterministic)
            
            # Take step
            next_state, reward, done, _ = env.step(action)
            
            # Render
            env.render()
            time.sleep(0.01)  # Slow down rendering
            
            # Log info
            episode_reward += reward
            steps += 1
            state = next_state
            
            # Consider success if episode ends with positive reward
            if done and reward >= 90:
                success_count += 1
        
        print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, Steps = {steps}")
        total_reward += episode_reward
    
    print(f"\nResults with {mode} policy:")
    print(f"Average reward over {num_episodes} episodes: {total_reward/num_episodes:.2f}")
    print(f"Success rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained SAC agent on MountainCarContinuous")
    parser.add_argument("--model", type=str, default="./results/sac_mountaincar", 
                      help="Path to model files (without _actor, _critic suffixes)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions instead of deterministic")
    
    args = parser.parse_args()
    
    # Test deterministic policy first
    if not args.stochastic:
        print("Testing deterministic policy...")
        test_agent(args.model, args.episodes, deterministic=True)
        
        # Ask if user wants to see stochastic policy too
        response = input("\nDo you want to test stochastic policy as well? (y/n): ")
        if response.lower() == 'y':
            print("\nTesting stochastic policy...")
            test_agent(args.model, args.episodes, deterministic=False)
    else:
        # Just test stochastic policy
        print("Testing stochastic policy...")
        test_agent(args.model, args.episodes, deterministic=False)