import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from env import EnvWrapper
from ppo import ImprovedPPO
from utils import plot_training_stats
from test import test_trained_model

# Make sure this matches your actual implementation or import it correctly
from gail import Discriminator as GAILDiscriminator

def train_ppo_with_irl(args, num_steps=200000, max_steps_per_episode=500,
                       update_frequency=2048, num_updates=10):
    """
    Train PPO using a reward function learned via GAIL's discriminator.
    """
    env = EnvWrapper(args.env, render=args.visualize)

    # PPO agent
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

    # Load GAIL discriminator
    gail_ckpt = torch.load("models/gail_cartpole.pth")
    discriminator = GAILDiscriminator(env.state_dim, env.action_dim)
    discriminator.load_state_dict(gail_ckpt['discriminator_state_dict'])
    discriminator.eval()

    reward_history = []
    episode_lengths = []
    best_avg_length = 0

    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    collected_steps = 0
    total_steps = 0

    print("Starting PPO training with GAIL reward function...")

    while total_steps < num_steps:
        action, action_probs, state_tensor = ppo.select_action(state)

        next_state, env_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # IRL reward from GAIL discriminator
        state_input = torch.FloatTensor(state).unsqueeze(0)
        action_input = torch.tensor([[action]])
        with torch.no_grad():
            irl_reward = -torch.log(1 - discriminator(state_input, action_input) + 1e-10).item()

        # Store IRL reward
        ppo.store_transition(
            state_tensor,
            action,
            action_probs,
            irl_reward,
            next_state,
            done
        )

        collected_steps += 1
        total_steps += 1
        episode_reward += env_reward  # use env_reward here just for tracking
        episode_length += 1
        state = next_state

        if done or episode_length >= max_steps_per_episode:
            reward_history.append(episode_reward)
            episode_lengths.append(episode_length)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)

            if avg_length > best_avg_length and episode_count > 20:
                best_avg_length = avg_length
                os.makedirs("models", exist_ok=True)
                ppo.save("models/ppo_cartpole_best.pth")
                print(f"New best model saved with avg length: {best_avg_length:.1f}")

            if episode_count % 10 == 0:
                print(f"Episode {episode_count}, reward: {episode_reward:.2f}, length: {episode_length}, "
                      f"avg length: {avg_length:.1f}, total steps: {total_steps}")

            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1

            if avg_length > 475 and episode_count > 100:
                print(f"Environment solved in {episode_count} episodes! Avg length: {avg_length:.1f}")
                break

        if collected_steps >= update_frequency:
            print(f"Updating PPO: {num_updates} updates after {collected_steps} steps")
            ppo.update(num_updates)
            collected_steps = 0

            if total_steps % 50000 == 0:
                os.makedirs("models", exist_ok=True)
                ppo.save(f"models/ppo_cartpole_steps_{total_steps}.pth")

    print("Training complete!")

    ppo.save(f"models/ppo_cartpole_irl_final.pth")
    env.close()
    return ppo, reward_history, episode_lengths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO with GAIL reward")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Train or test mode')
    parser.add_argument('--model', type=str, default='models/ppo_cartpole_best.pth', help='Path to saved model')
    parser.add_argument('--total_steps', type=int, default=200000, help='Total training steps')
    parser.add_argument('--update_frequency', type=int, default=2048, help='Steps before PPO update')
    parser.add_argument('--num_updates', type=int, default=10, help='Number of PPO updates')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes for testing')
    parser.add_argument('--visualize', action='store_true', help='Render test episodes')
    parser.add_argument('--env', type=str, default='CartPole-v1', help='Gym environment name')

    args = parser.parse_args()

    if args.mode == 'train':
        ppo, rewards, lengths = train_ppo_with_irl(
            args,
            num_steps=args.total_steps,
            update_frequency=args.update_frequency,
            num_updates=args.num_updates
        )
        plot_training_stats(rewards, lengths, show_plot=False, save_plot=True)
        print("Test best model after training...")
        test_trained_model(args.env, 'models/ppo_cartpole_best.pth', num_episodes=3, visualize=True)

    elif args.mode == 'test':
        test_trained_model(args.env, args.model, num_episodes=args.episodes, visualize=args.visualize)
