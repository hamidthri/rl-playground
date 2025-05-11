import numpy as np
import torch
import argparse
import os
import time
from env import MountainCarEnvWrapper
from sac import SAC
import matplotlib.pyplot as plt
from collections import deque

# Add reward shaping function
def reward_shaping(state, action, reward, next_state, done):
    """
    Shape the reward to encourage exploration and progress toward the goal
    
    In MountainCar:
    - position ranges roughly from -1.2 to 0.6
    - velocity ranges from -0.07 to 0.07
    - goal position is at position > 0.45
    """
    # Base reward from environment
    shaped_reward = reward
    
    position, velocity = next_state
    
    # Reward for moving to the right (toward goal)
    shaped_reward += position * 0.1
    
    # Reward for having positive velocity (moving right)
    if velocity > 0:
        shaped_reward += velocity * 10.0
    
    # Extra reward for reaching higher positions
    if position > 0:
        shaped_reward += position * 0.5
    
    # Large reward for getting close to goal
    if position > 0.4:
        shaped_reward += 1.0
    
    # Penalize staying at the bottom
    if -0.6 < position < -0.4:
        shaped_reward -= 0.1
    
    return shaped_reward

def evaluate_policy(policy, env, eval_episodes=10, render=False, deterministic=True):
    """
    Evaluate the policy for a certain number of episodes and return the average reward.
    If render is True, the environment will be rendered during evaluation.
    If deterministic is True, the policy will use its mean action instead of sampling.
    """
    avg_reward = 0.
    all_episode_rewards = []
    successes = 0
    
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        max_position = -1.2  # Track max position reached
        
        while not done:
            if render:
                env.render()
                time.sleep(0.01)  # Add small delay to make rendering visible
            
            # Select action (deterministic or stochastic)
            action = policy.select_action(state, evaluate=deterministic)
            
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Track maximum position
            max_position = max(max_position, next_state[0])
            
            state = next_state
        
        # Consider episode successful if max position is close to goal
        if max_position > 0.45:
            successes += 1
        
        print(f"Episode finished with reward: {episode_reward:.3f} in {episode_steps} steps, max pos: {max_position:.3f}")
        all_episode_rewards.append(episode_reward)
        avg_reward += episode_reward
    
    avg_reward /= eval_episodes
    success_rate = successes / eval_episodes
    
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Min reward: {min(all_episode_rewards):.3f}, Max reward: {max(all_episode_rewards):.3f}")
    print("---------------------------------------")
    return avg_reward

def load_and_evaluate(args, render=True):
    """
    Load a trained model and evaluate it with visualization.
    Tries both deterministic and stochastic approaches.
    """
    # Initialize environment with rendering
    env = MountainCarEnvWrapper(render_mode="human" if render else None)
    
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
    
    # Load the saved model
    model_path = "./results/sac_mountaincar"
    agent.load(model_path)
    
    print(f"Loaded model from {model_path}")
    
    # Evaluate with deterministic policy
    print("\nEvaluating with deterministic policy:")
    evaluate_policy(agent, env, eval_episodes=5, render=render, deterministic=True)
    
    # Give some time between evaluations if rendering
    if render:
        time.sleep(2)
    
    # Evaluate with stochastic policy
    print("\nEvaluating with stochastic policy:")
    evaluate_policy(agent, env, eval_episodes=5, render=render, deterministic=False)
    
    env.close()

def main(args):
    # Create output directory
    if not os.path.exists("results"):
        os.makedirs("results")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize environment
    env = MountainCarEnvWrapper(render_mode=args.render_mode)
    eval_env = MountainCarEnvWrapper()
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action
    
    print(f"Environment: MountainCarContinuous-v0")
    print(f"State dimensions: {state_dim}")
    print(f"Action dimensions: {action_dim}")
    print(f"Action range: [{-max_action}, {max_action}]")
    
    # Initialize SAC agent with improved parameters
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
    
    # Additional logging for debugging
    max_positions = []
    
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
                    eval_reward = evaluate_policy(agent, eval_env, deterministic=True)
                    evaluations.append((total_timesteps, eval_reward))
                    
                    # Save the policy
                    if args.save_model:
                        agent.save(f"./results/sac_mountaincar")
            
            # Reset environment
            state = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            max_position = -1.2  # Track maximum position in episode
        
        # Collect data with some initial random exploration
        if total_timesteps < args.start_timesteps:
            action = env.sample_action()
        else:
            action = agent.select_action(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(action)
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1
        
        # Apply reward shaping
        shaped_reward = reward_shaping(state, action, reward, next_state, done)
        episode_reward += shaped_reward
        
        # Track maximum position
        max_position = max(max_position, next_state[0])
        
        # Store transition in replay buffer with shaped reward
        agent.store_transition(state, action, shaped_reward, next_state, float(done))
        
        # Update agent
        if total_timesteps >= args.start_timesteps:
            for _ in range(args.updates_per_step):
                agent.update_parameters(args.batch_size)
        
        # Move to the next state
        state = next_state
        
        # End episode early if car reaches goal position
        if state[0] >= 0.45:
            print(f"Success! Car reached position {state[0]:.3f} at step {episode_timesteps}")
            if not done:  # Only add bonus reward if the environment didn't already end
                # Add bonus reward for reaching goal
                agent.store_transition(state, action, 100.0, next_state, 1.0)
                episode_reward += 100.0
                done = True
        
        # Log max position every 1000 steps
        if total_timesteps % 1000 == 0:
            max_positions.append((total_timesteps, max_position))
                
        # End training if max timesteps reached
        if total_timesteps >= args.max_timesteps:
            break
    
    # Final evaluation
    evaluate_policy(agent, eval_env)
    
    # Save the final model
    if args.save_model:
        agent.save(f"./results/sac_mountaincar_final")
    
    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot([t for t, _ in training_rewards], [r for _, r in training_rewards], alpha=0.3, label='Episode Rewards')
    plt.plot([t for t, _ in evaluations], [r for _, r in evaluations], label='Evaluation Rewards')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.title('SAC Training on MountainCarContinuous-v0')
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
    plt.title('SAC Average Training Reward on MountainCarContinuous-v0')
    plt.savefig("./results/average_reward.png")
    plt.close()
    
    # Plot maximum position
    plt.figure(figsize=(10, 5))
    plt.plot([t for t, _ in max_positions], [p for _, p in max_positions])
    plt.axhline(y=0.45, color='r', linestyle='--', label='Goal Position')
    plt.xlabel('Timesteps')
    plt.ylabel('Maximum Position')
    plt.title('Maximum Position Reached During Training')
    plt.legend()
    plt.savefig("./results/max_position.png")
    plt.close()

def test_model(args):
    """Run testing with visualization"""
    load_and_evaluate(args, render=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)              # Random seed
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--alpha", default=0.2, type=float)         # Temperature parameter (if not automatic)
    parser.add_argument("--automatic_entropy_tuning", default=True, action='store_true')  # Automatic entropy tuning
    parser.add_argument("--hidden_dim", default=256, type=int)      # Hidden dimension for networks
    parser.add_argument("--lr", default=3e-4, type=float)           # Learning rate
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size
    parser.add_argument("--max_timesteps", default=200000, type=int)  # Max time steps for training
    parser.add_argument("--start_timesteps", default=5000, type=int)  # Initial random exploration steps
    parser.add_argument("--updates_per_step", default=1, type=int)  # Updates per environment step
    parser.add_argument("--eval_freq", default=5000, type=int)      # How often to evaluate
    parser.add_argument("--buffer_size", default=1000000, type=int) # Replay buffer size
    parser.add_argument("--save_model", default=True, action='store_true')    # Whether to save model
    parser.add_argument("--render_mode", default=None, type=str)    # Render mode (human, rgb_array, None)
    parser.add_argument("--mode", default="train", choices=["train", "test"], 
                      help="Whether to train a new model or test an existing one")
    
    args = parser.parse_args()
    
    
        
        