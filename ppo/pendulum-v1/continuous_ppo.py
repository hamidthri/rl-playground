import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    """Basic residual block with layer normalization"""
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        x += residual  # Skip connection
        return torch.relu(x)


# Actor network for continuous actions with improved architecture
class ContinuousActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, action_scale=1.0, num_res_blocks=2):
        super(ContinuousActorNetwork, self).__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Initial layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Output mean of the action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        nn.init.xavier_uniform_(self.mean.weight, gain=0.01)
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)
        
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Get mean of the distribution
        action_mean = torch.tanh(self.mean(x)) * self.action_scale
        action_log_std = self.log_std.expand_as(action_mean)
        action_log_std = torch.clamp(action_log_std, -2, 2)
        
        return action_mean, action_log_std
    
    def get_dist(self, state):
        """Get action distribution for a given state"""
        action_mean, action_log_std = self.forward(state)
        action_std = torch.exp(action_log_std)
        
        # Create normal distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist, action_mean
    
    def sample(self, state):
        """Sample action from the distribution"""
        dist, action_mean = self.get_dist(state)
        action = dist.rsample()  # Use reparameterization trick for better gradients
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, action_log_prob
    
    def evaluate(self, state, action):
        """Evaluate action log probability and entropy"""
        dist, _ = self.get_dist(state)
        
        action_log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return action_log_prob, entropy

# Improved Critic network with layer normalization
class ContinuousCriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256, num_res_blocks=2):
        super(ContinuousCriticNetwork, self).__init__()
        
        # Initial layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_res_blocks)
        ])
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.01)
        
    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        
        # Pass through residual blocks
        for block in self.res_blocks:
            x = block(x)
            
        value = self.fc_out(x)
        return value

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, action_log_prob, reward, next_state, done):
        self.memory.append((state, action, action_log_prob, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def clear(self):
        self.memory.clear()
    
    def __len__(self):
        return len(self.memory)

class ContinuousPPO:
    def __init__(self, state_dim, action_dim, action_scale=1.0, 
                 lr_actor=3e-4, lr_critic=3e-4, gamma=0.99,
                 gae_lambda=0.95, K_epochs=10, eps_clip=0.2,
                 replay_capacity=10000, batch_size=256,
                 num_res_blocks=2):  # New parameter
        """
        Initialize PPO algorithm with continuous actions
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_scale: Maximum action magnitude
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            K_epochs: Number of epochs for policy update
            eps_clip: Clip parameter for PPO
            replay_capacity: Capacity of replay memory
            batch_size: Batch size for training
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Initialize actor and critic networks
        self.actor = ContinuousActorNetwork(
                state_dim, action_dim, 
                action_scale=action_scale,
                num_res_blocks=num_res_blocks
            ).to(device)
        
        self.critic = ContinuousCriticNetwork(
                state_dim,
                num_res_blocks=num_res_blocks
            ).to(device)
        
        # Old actor for PPO update (target network)
        self.actor_old = ContinuousActorNetwork(
                state_dim, action_dim,
                action_scale=action_scale,
                num_res_blocks=num_res_blocks
            ).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Initialize optimizers with learning rate schedule
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Learning rate schedulers for better convergence
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=2000, gamma=0.95)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=2000, gamma=0.95)
        
        # Initialize replay memory
        self.memory = ReplayMemory(capacity=replay_capacity)
        
        # MSE loss for value function
        self.MseLoss = nn.MSELoss()
        
        # Add entropy coefficient for better exploration
        self.entropy_coef = 0.01
    
    def select_action(self, state):
        """
        Select a continuous action based on current policy
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action vector
            action_log_prob: Log probability of the action
            state_tensor: State tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        
        # Ensure state is 2D for batch processing
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action, action_log_prob = self.actor_old.sample(state)
        
        return action.cpu().numpy().flatten(), action_log_prob, state
    
    def store_transition(self, state, action, action_log_prob, reward, next_state, done):
        """Store a transition in replay memory"""
        # Convert to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(device)
        
        # Ensure states are 2D for batch processing
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Store transition in memory
        self.memory.push(state, action, action_log_prob, reward, next_state, done)
    
    def normalize_rewards(self, rewards):
        """Normalize rewards for better stability"""
        return (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    
    def calculate_gae(self, rewards, values, next_values, dones):
        """Calculate Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        returns = advantages + values
        
        return returns, advantages
    
    def update(self, num_updates=1):
        """Update policy using the PPO algorithm with continuous actions"""
        # Return if memory is too small
        if len(self.memory) < self.batch_size:
            print(f"Not enough samples in memory ({len(self.memory)}/{self.batch_size}). Skipping update.")
            return
        
        # Calculate average reward for debugging
        avg_reward = 0
        for _, _, _, reward, _, _ in self.memory.memory:
            avg_reward += reward
        avg_reward /= len(self.memory)
        
        # Perform multiple updates
        for _ in range(num_updates):
            # Sample batch from memory
            batch = self.memory.sample(self.batch_size)
            
            # Unpack batch
            states = torch.cat([x[0] for x in batch])
            actions = torch.cat([x[1] for x in batch])
            old_action_log_probs = torch.cat([x[2] for x in batch])
            rewards = torch.FloatTensor([x[3] for x in batch]).unsqueeze(1).to(device)
            next_states = torch.cat([x[4] for x in batch])
            dones = torch.FloatTensor([x[5] for x in batch]).unsqueeze(1).to(device)
            
            # Normalize rewards for better training stability
            rewards = self.normalize_rewards(rewards)
            
            # Compute value estimates
            with torch.no_grad():
                values = self.critic(states)
                next_values = self.critic(next_states)
                
            # Calculate returns and advantages using GAE
            returns, advantages = self.calculate_gae(rewards, values, next_values, dones)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Optimize policy for K epochs
            for _ in range(self.K_epochs):
                # Get current action log probabilities and values
                current_action_log_probs, entropy = self.actor.evaluate(states, actions)
                current_values = self.critic(states)
                
                # Calculate ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(current_action_log_probs - old_action_log_probs.detach())
                
                # Calculate surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                # PPO loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy term for exploration with adaptive coefficient
                actor_loss = actor_loss - self.entropy_coef * entropy.mean()
                
                # Value loss with clipping for stability
                value_pred_clipped = values + torch.clamp(current_values - values, -self.eps_clip, self.eps_clip)
                value_loss_1 = self.MseLoss(current_values, returns.detach())
                value_loss_2 = self.MseLoss(value_pred_clipped, returns.detach())
                value_loss = torch.max(value_loss_1, value_loss_2)
                
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                
                # Update critic
                self.optimizer_critic.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
        
        # Update learning rates
        self.scheduler_actor.step()
        self.scheduler_critic.step()
        
        # Gradually reduce entropy coefficient for less exploration over time
        self.entropy_coef = max(0.001, self.entropy_coef * 0.995)
        
        # Copy new weights to old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Clear memory after updates to avoid reusing old samples
        self.memory.clear()
    
    def save(self, filename):
        """Save policy weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_old': self.actor_old.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
            'scheduler_actor': self.scheduler_actor.state_dict(),
            'scheduler_critic': self.scheduler_critic.state_dict(),
            'entropy_coef': self.entropy_coef
        }, filename)
    
    def load(self, filename):
        """Load policy weights"""
        checkpoint = torch.load(filename, map_location=device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_old.load_state_dict(checkpoint['actor_old'])
        if 'optimizer_actor' in checkpoint:
            self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        if 'optimizer_critic' in checkpoint:
            self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])
        if 'scheduler_actor' in checkpoint:
            self.scheduler_actor.load_state_dict(checkpoint['scheduler_actor'])
        if 'scheduler_critic' in checkpoint:
            self.scheduler_critic.load_state_dict(checkpoint['scheduler_critic'])
        if 'entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['entropy_coef']