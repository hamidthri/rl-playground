import numpy as np
import torch
import argparse
import os
import time
from env import PendulumEnvWrapper
from sac import SAC
import matplotlib.pyplot as plt
from collections import deque

def evaluate_policy(policy, env, eval_episodes=10):
    """
    Evaluate the policy for a certain number of episodes and return the average reward.
    """
    avg_reward = 0.
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy.select_action(state)
            next_state, reward, done, _ = env.step(action)
            avg_reward += reward
            state = next_state
    
    avg_reward /= eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def main(args):
    # Create output directory
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize environment
    env = PendulumEnvWrapper(render_mode=args.render_mode)
    eval_env = PendulumEnvWrapper()
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action
    
    # Initialize SAC agent
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        automatic_entropy_tuning=args.automatic_entropy_tuning,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        buffer_size=args.buffer_size
    )
    
    # Set up logging
    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    done = True
    
    # Initialize lists for logging
    evaluations = []
    training_rewards = []
    avg_rewards = deque(maxlen=100)
    
    # Training loop
    while total_timesteps < args.max_timesteps:
        if done:
            if total_timesteps != 0:
                print(f"Total T: {total_timesteps} Episode Num: {episode_num} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                avg_rewards.append(episode_reward)
                training_rewards.append((total_timesteps, episode_reward))
                
                # Perform evaluation if it's time
                if timesteps_since_eval >= args.eval_freq:
                    timesteps_since_eval %= args.eval_freq
                    eval_reward = evaluate_policy(agent, eval_env)
                    evaluations.append((total_timesteps, eval_reward))
                    
                    # Save the policy
                    if args.save_model:
                        agent.save(f"./results/sac_pendulum_{total_timesteps}")
            
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Collect data with some initial random exploration
        if total_timesteps < args.start_timesteps:
            action = env.sample_action()
        else:
            action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        episode_reward += reward
        
        # Store transition in replay buffer
        agent.store_transition(state, action, reward, next_state, float(done))
        
        # Update agent
        if total_timesteps >= args.start_timesteps:
            for _ in range(args.updates_per_step):
                agent.update_parameters(args.batch_size)
        
        # Move to the next state
        state = next_state
        
        # End training if max timesteps reached
        if total_timesteps >= args.max_timesteps:
            break
    
    # Final evaluation
    evaluate_policy(agent, eval_env)
    
    # Save the final model
    if args.save_model:
        agent.save(f"./results/sac_pendulum_final")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot([t for t, _ in training_rewards], [r for _, r in training_rewards], alpha=0.3, label='Episode Rewards')
    plt.plot([t for t, _ in evaluations], [r for _, r in evaluations], label='Evaluation Rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('SAC Training on Pendulum-v1')
    plt.legend()
    plt.savefig("./results/training_curve.png")
    plt.close()
    
    # Plot the average training reward
    avg_reward_list = []
    window_size = 100
    for i in range(len(training_rewards)):
        if i < window_size:
            avg_reward_list.append(np.mean([r for _, r in training_rewards[:i+1]]))
        else:
            avg_reward_list.append(np.mean([r for _, r in training_rewards[i-window_size+1:i+1]]))
    
    plt.figure(figsize=(10, 5))
    plt.plot([t for t, _ in training_rewards], avg_reward_list)
    plt.xlabel('Timesteps')
    plt.ylabel('Average Reward (100 episodes)')
    plt.title('SAC Average Training Reward on Pendulum-v1')
    plt.savefig("./results/average_reward.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)              # Random seed
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--alpha", default=0.2, type=float)         # Temperature parameter (if not automatic)
    parser.add_argument("--automatic_entropy_tuning", default=True, type=bool)  # Automatic entropy tuning
    parser.add_argument("--hidden_dim", default=256, type=int)      # Hidden dimension for networks
    parser.add_argument("--lr", default=3e-4, type=float)           # Learning rate
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size
    parser.add_argument("--max_timesteps", default=1000000, type=int)  # Max time steps for training
    parser.add_argument("--start_timesteps", default=10000, type=int)  # Initial random exploration steps
    parser.add_argument("--updates_per_step", default=1, type=int)  # Updates per environment step
    parser.add_argument("--eval_freq", default=5000, type=int)      # How often to evaluate
    parser.add_argument("--buffer_size", default=1000000, type=int) # Replay buffer size
    parser.add_argument("--save_model", default=True, type=bool)    # Whether to save model
    parser.add_argument("--render_mode", default=None, type=str)    # Render mode (human, rgb_array, None)
    
    args = parser.parse_args()
    
    main(args)