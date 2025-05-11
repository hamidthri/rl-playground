import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from rl.ppo.cartpole.actor import ActorNetwork

import argparse
import os
import time
import tqdm
from tqdm import trange


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.
    Distinguishes between expert demonstrations and agent rollouts.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        # Input is state-action pair
        input_dim = state_dim + action_dim
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output is probability of being from expert
        )
        
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, states, actions):
        """
        Forward pass through the discriminator.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size, action_dim] or [batch_size] for discrete actions
            
        Returns:
            Probability that each state-action pair is from expert
        """
        # For discrete actions, convert to one-hot
        if actions.dtype == torch.int64 or actions.dtype == torch.long:
            # Get the batch size
            batch_size = actions.size(0)
            
            # Create one-hot tensor with proper size
            actions_one_hot = torch.zeros(batch_size, self.action_dim, device=actions.device)
            
            # Ensure actions is the right shape before using scatter_
            if actions.dim() == 1:
                # If actions is [batch_size], unsqueeze to [batch_size, 1]
                actions = actions.unsqueeze(1)
                
            # Fill one-hot tensor
            actions_one_hot.scatter_(1, actions, 1)
            actions = actions_one_hot
        
        # Concatenate state and action
        state_action = torch.cat([states, actions], dim=1)
        return self.model(state_action)

class PolicyNetwork(nn.Module):
    """Policy network for GAIL agent."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Policy head outputs action probabilities
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value head estimates state value
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, states):
        """Forward pass to get action probabilities and value estimates."""
        features = self.trunk(states)
        action_probs = self.policy(features)
        values = self.value(features)
        return action_probs, values
    
    def act(self, state, deterministic=False):
        """Sample an action from the policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = self.forward(state)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
            return action.item()


class PPOBuffer:
    """Buffer for storing trajectories collected during training."""
    def __init__(self, state_dim, action_dim, buffer_size, device):
        self.device = device
        
        # Initialize buffers
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, 1), dtype=torch.long, device=device)
        self.logprobs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.values = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        
        self.idx = 0
        self.buffer_size = buffer_size
        self.full = False
    
    def add(self, state, action, logprob, reward, done, value):
        """Add a transition to the buffer."""
        self.states[self.idx] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.idx] = torch.as_tensor(action, dtype=torch.long, device=self.device)
        self.logprobs[self.idx] = logprob
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.values[self.idx] = value
        
        self.idx = (self.idx + 1) % self.buffer_size
        if self.idx == 0:
            self.full = True
    
    def get(self):
        """Get all data from the buffer."""
        if self.full:
            end_idx = self.buffer_size
        else:
            end_idx = self.idx
            
        return (
            self.states[:end_idx],
            self.actions[:end_idx],
            self.logprobs[:end_idx],
            self.rewards[:end_idx],
            self.dones[:end_idx],
            self.values[:end_idx]
        )
    
    def clear(self):
        """Reset the buffer."""
        self.idx = 0
        self.full = False


class GAIL:
    """
    Generative Adversarial Imitation Learning implementation.
    Learns a policy from expert demonstrations using adversarial training.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        expert_states,
        expert_actions,
        lr_policy=3e-4,
        lr_discriminator=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        entropy_beta=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=10,
        batch_size=64,
        buffer_size=2048,
        d_steps=1,
        hidden_dim=64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.entropy_beta = entropy_beta
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.d_steps = d_steps
        
        # Convert expert data to tensors
        self.expert_states = torch.FloatTensor(expert_states).to(device)
        self.expert_actions = torch.LongTensor(expert_actions).to(device)
        
        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.discriminator = Discriminator(state_dim, action_dim, hidden_dim).to(device)
        
        # Initialize optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.discriminator_optim = optim.Adam(self.discriminator.parameters(), lr=lr_discriminator)
        
        # Initialize buffer
        self.buffer = PPOBuffer(state_dim, action_dim, buffer_size, device)
        
        # Track metrics
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.discriminator_losses = []
        self.rewards = []
    
    def get_action(self, state, deterministic=False):
        """Sample an action from the policy."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_probs, value = self.policy(state.unsqueeze(0))
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                
            logprob = torch.log(action_probs.squeeze(0)[action.item()] + 1e-10)
            
            return action.item(), logprob.item(), value.item()
    
    def compute_gae(self, next_value, rewards, dones, values, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
                
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_val * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        return advantages, returns
    
    def update_discriminator(self, agent_states, agent_actions):
        """Update the discriminator network."""
        # Prepare agent and expert data
        agent_states = agent_states.to(self.device)
        agent_actions = agent_actions.to(self.device)
        
        # Create one-hot encoded actions for agent
        agent_actions_one_hot = torch.zeros(agent_actions.size(0), self.action_dim, device=self.device)
        agent_actions_one_hot.scatter_(1, agent_actions, 1)
        
        # Create one-hot encoded actions for expert
        expert_actions_unsqueezed = self.expert_actions.unsqueeze(1)
        expert_actions_one_hot = torch.zeros(
            self.expert_actions.size(0), self.action_dim, device=self.device
        )
        expert_actions_one_hot.scatter_(1, expert_actions_unsqueezed, 1)
        
        # Randomly sample expert demonstrations to match agent batch size
        expert_indices = torch.randperm(self.expert_states.size(0))[:agent_states.size(0)]
        expert_states_batch = self.expert_states[expert_indices]
        expert_actions_batch = expert_actions_one_hot[expert_indices]
        
        for _ in range(self.d_steps):
            # Discriminator predictions
            expert_preds = self.discriminator(expert_states_batch, expert_actions_batch)
            agent_preds = self.discriminator(agent_states, agent_actions_one_hot)
            
            # Binary cross entropy loss
            expert_loss = -torch.log(expert_preds + 1e-10).mean()
            agent_loss = -torch.log(1 - agent_preds + 1e-10).mean()
            discriminator_loss = expert_loss + agent_loss
            
            # Update discriminator
            self.discriminator_optim.zero_grad()
            discriminator_loss.backward()
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
            self.discriminator_optim.step()
            
            self.discriminator_losses.append(discriminator_loss.item())
        
        # Return rewards for the collected trajectories (negative logits from discriminator)
        with torch.no_grad():
            agent_preds = self.discriminator(agent_states, agent_actions_one_hot)
            rewards = -torch.log(1 - agent_preds + 1e-10)
            
        return rewards
    
    def update_policy(self, states, actions, old_logprobs, rewards, dones, old_values, next_value):
        """Update the policy network using PPO."""
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value, rewards, dones, old_values, self.gamma)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        for _ in range(self.ppo_epochs):
            # Create mini-batches
            batch_size = min(self.batch_size, states.size(0))
            indices = torch.randperm(states.size(0))
            
            for start_idx in range(0, states.size(0), batch_size):
                end_idx = min(start_idx + batch_size, states.size(0))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Forward pass
                action_probs, values = self.policy(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                
                # Calculate new log probabilities
                new_logprobs = dist.log_prob(batch_actions.squeeze(-1)).unsqueeze(-1)
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Policy loss with clipping
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - batch_returns) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_beta * entropy
                
                # Optimize
                self.policy_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optim.step()
                
                # Record metrics
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy.item())
    
    def train_step(self, env, steps_per_update=2048):
        """Perform one training iteration (collect trajectories and update networks)."""
        # Collect trajectories
        states, actions, logprobs, rewards, dones, values = [], [], [], [], [], []
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while step_count < steps_per_update:
            # Get action from policy
            action, logprob, value = self.get_action(state)
            
            # Take step in environment
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            values.append(value)
            dones.append(done)
            
            # Update state
            state = next_state
            step_count += 1
            total_reward += env_reward
            
            # Reset environment if done
            if done:
                state, _ = env.reset()
                self.rewards.append(total_reward)
                total_reward = 0
        
        # Get value of last state for bootstrapping
        if not done:
            _, _, next_value = self.get_action(state)
        else:
            next_value = 0
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(-1).to(self.device)
        logprobs_tensor = torch.FloatTensor(logprobs).unsqueeze(-1).to(self.device)
        values_tensor = torch.FloatTensor(values).unsqueeze(-1).to(self.device)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1).to(self.device)
        
        # Update discriminator and get rewards
        rewards_tensor = self.update_discriminator(states_tensor, actions_tensor)
        
        # Update policy
        self.update_policy(
            states_tensor,
            actions_tensor,
            logprobs_tensor,
            rewards_tensor,
            dones_tensor,
            values_tensor,
            next_value
        )
    
    def train(self, env, num_iterations=100, steps_per_update=2048, eval_interval=10):
        """Train the agent for a specified number of iterations."""
        eval_rewards = []
        
        for iteration in range(num_iterations):
            start_time = time.time()
            
            # Train for one iteration
            self.train_step(env, steps_per_update)
            
            end_time = time.time()
            iteration_time = end_time - start_time

            # Log current metrics
            policy_loss = np.mean(self.policy_losses[-self.ppo_epochs:]) if self.policy_losses else 0
            value_loss = np.mean(self.value_losses[-self.ppo_epochs:]) if self.value_losses else 0
            disc_loss = np.mean(self.discriminator_losses[-self.d_steps:]) if self.discriminator_losses else 0
            episode_rewards = self.rewards[-10:] if self.rewards else [0]

            print(f"[{iteration+1:03d}/{num_iterations}] "
                f"Policy Loss: {policy_loss:.4f}, "
                f"Value Loss: {value_loss:.4f}, "
                f"Disc Loss: {disc_loss:.4f}, "
                f"EpReward (mean ± std): {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}, "
                f"Time: {iteration_time:.2f}s")

            if (iteration + 1) % eval_interval == 0:
                avg_reward = self.evaluate(env, num_episodes=5)
                eval_rewards.append(avg_reward)
                print(f"→ Evaluation Avg Reward @Iter {iteration + 1}: {avg_reward:.2f}")
        
        return eval_rewards
    
    def evaluate(self, env, num_episodes=10):
        """Evaluate the current policy without exploration."""
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.policy.act(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
        
        return np.mean(rewards)
    
    def save(self, path):
        """Save model checkpoints as .pth file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Ensure the path ends with .pth
        if not path.endswith('.pth'):
            path = path + '.pth'
        
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_optimizer_state_dict': self.policy_optim.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optim.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            # Add any other hyperparameters you want to save
        }, path)
        
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model from .pth file"""
        checkpoint = torch.load(path)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.discriminator_optim.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        
        print(f"Model loaded from {path}")
    
    def render_policy(self, env_name="CartPole-v1", num_episodes=3):
        """Render the current policy."""
        render_env = gym.make(env_name, render_mode="human")  # Ensure this is correct

        for episode in range(num_episodes):
            state, _ = render_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.policy.act(state, deterministic=True)
                state, reward, terminated, truncated, _ = render_env.step(action)
                episode_reward += reward
                done = terminated or truncated
                render_env.render()  # Explicit render call

            print(f"Episode {episode + 1}: Reward = {episode_reward}")

        render_env.close()



def load_expert_model(state_dim, action_dim, path):
    """Load a pre-trained expert model."""
    checkpoint = torch.load(path)
    model = ActorNetwork(state_dim, action_dim, hidden_dim=64)
    model.load_state_dict(checkpoint["actor"])
    model.eval()
    return model


def collect_expert_demos(model, env_name="CartPole-v1", n_episodes=100, max_steps=500):
    """Collect expert demonstrations using a pre-trained model."""
    env = gym.make(env_name)
    all_states = []
    all_actions = []
    
    for _ in trange(n_episodes):
        states, actions = [], []
        obs, _ = env.reset()
        
        for _ in range(max_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = model(obs_tensor)
                action = logits.argmax(dim=-1).item()
            
            states.append(obs)
            actions.append(action)
            
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        if len(states) > 10:  # Only include trajectories of reasonable length
            all_states.extend(states)
            all_actions.extend(actions)
    
    print(f"Collected {len(all_states)} expert state-action pairs.")
    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.int64)


def plot_training_curves(gail_agent):
    """Plot training curves."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot policy loss
    ax[0, 0].plot(gail_agent.policy_losses)
    ax[0, 0].set_title('Policy Loss')
    ax[0, 0].set_xlabel('Update Steps')
    
    # Plot discriminator loss
    ax[0, 1].plot(gail_agent.discriminator_losses)
    ax[0, 1].set_title('Discriminator Loss')
    ax[0, 1].set_xlabel('Update Steps')
    
    # Plot value loss
    ax[1, 0].plot(gail_agent.value_losses)
    ax[1, 0].set_title('Value Loss')
    ax[1, 0].set_xlabel('Update Steps')
    
    # Plot episode rewards during training
    if gail_agent.rewards:
        ax[1, 1].plot(gail_agent.rewards)
        ax[1, 1].set_title('Episode Rewards')
        ax[1, 1].set_xlabel('Episodes')
    
    plt.tight_layout()
    plt.savefig('gail_training_curves.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="GAIL for CartPole")
    parser.add_argument("--ppo-path", type=str, default="/Users/htaheri/Documents/GitHub/rl-playground/rl/ppo/cartpole/models/ppo_cartpole_best.pth",
                        help="Path to PPO expert model")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of training iterations")
    parser.add_argument("--steps-per-update", type=int, default=2048,
                        help="Number of steps per update")
    parser.add_argument("--expert-demos", type=int, default=20,
                        help="Number of expert demonstrations to collect")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                        help="Gym environment")
    parser.add_argument("--render", action="store_true",
                        help="Render the trained policy (in test mode)")
    parser.add_argument("--cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Run training or test mode")
    parser.add_argument("--model-path", type=str, default="models/gail_cartpole.pth",
                        help="Path to save/load the model")

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Create environment and get dimensions
    env = gym.make(args.env)  # No render mode here


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load expert model
    print("Loading expert model...")
    expert_model = load_expert_model(state_dim, action_dim, args.ppo_path)

    # Collect expert demonstrations
    print(f"Collecting {args.expert_demos} expert demonstrations...")
    expert_states, expert_actions = collect_expert_demos(
        expert_model, args.env, n_episodes=args.expert_demos
    )

    # Initialize GAIL agent
    gail_agent = GAIL(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        expert_states=expert_states,
        expert_actions=expert_actions,
        buffer_size=args.steps_per_update
    )

    if args.mode == "train":
        print(f"Training GAIL for {args.iterations} iterations...")
        eval_rewards = gail_agent.train(
            env,
            num_iterations=args.iterations,
            steps_per_update=args.steps_per_update
        )

        # Plot training curves
        plot_training_curves(gail_agent)

        # Plot evaluation rewards
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(10, args.iterations + 1, 10), eval_rewards)
        plt.xlabel('Iterations')
        plt.ylabel('Average Reward')
        plt.title('GAIL Evaluation Rewards')
        plt.savefig('gail_eval_rewards.png')
        plt.show()

        # Save the trained model
        gail_agent.save(args.model_path)

        # Final evaluation
        print("Final evaluation...")
        final_reward = gail_agent.evaluate(env, num_episodes=10)
        print(f"Final average reward: {final_reward:.2f}")

    elif args.mode == "test":
        # Load model
        gail_agent.load(args.model_path)

        # Evaluate or render policy
        if args.render:
            print("Rendering trained policy...")
            gail_agent.render_policy(args.env)  # Pass the env name
        else:
            avg_reward = gail_agent.evaluate(env)
            print(f"Evaluation average reward: {avg_reward:.2f}")

    env.close()



if __name__ == "__main__":
    main()