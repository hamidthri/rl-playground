import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import math
from rl.imitation.irl.train_ppo_with_irl.actor import ActorNetwork
from rl.imitation.irl.train_ppo_with_irl.critic import CriticNetwork
from rl.imitation.irl.train_ppo_with_irl.utils import ReplayMemory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImprovedPPO:
    def __init__(self, state_dim, action_dim, lr_actor=0.0003, lr_critic=0.001, gamma=0.99, 
                 K_epochs=10, eps_clip=0.2, replay_capacity=10000, batch_size=64):
        """
        Initialize PPO algorithm with discrete actions for CartPole
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (2 for CartPole - left/right)
            lr_actor: Learning rate for actor network
            lr_critic: Learning rate for critic network
            gamma: Discount factor
            K_epochs: Number of epochs for policy update
            eps_clip: Clip parameter for PPO
            replay_capacity: Capacity of replay memory
            batch_size: Batch size for training
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.action_dim = action_dim
        
        # Initialize actor and critic with simpler architecture
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # Old actor for PPO update (target network)
        self.actor_old = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        
        # Initialize optimizers with smaller learning rates for stability
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize replay memory
        self.memory = ReplayMemory(capacity=replay_capacity)
        
        # MSE loss for value function
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        """
        Select a discrete action based on current policy
        
        Args:
            state: Current state
            
        Returns:
            action: Selected action (0 or 1)
            action_probs: Action probabilities
            state_tensor: State tensor
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        
        # Ensure state is 2D for batch processing
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.actor_old(state)
            action_probs = torch.softmax(action_logits, dim=-1)
            
            # Sample action from the distribution
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        
        return action.item(), action_probs, state
    
    def store_transition(self, state, action, action_probs, reward, next_state, done):
        """Store a transition in replay memory"""
        # Convert to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        if isinstance(next_state, np.ndarray):
            next_state = torch.FloatTensor(next_state).to(device)
        
        # Ensure states are 2D for batch processing
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(next_state.shape) == 1:
            next_state = next_state.unsqueeze(0)
        
        # Store transition in memory
        self.memory.push(state, action, action_probs, reward, next_state, done)
    
    def update(self, num_updates=1):
        """Update policy using the PPO algorithm with discrete actions"""
        # Return if memory is too small
        if len(self.memory) < self.batch_size:
            print(f"Not enough samples in memory ({len(self.memory)}/{self.batch_size}). Skipping update.")
            return
        
        # Calculate returns more accurately with n-step returns
        # Process all trajectories in memory to calculate returns properly
        processed_memory = []
        
        # Perform multiple updates
        for _ in range(num_updates):
            # Sample batch from memory
            batch = self.memory.sample(self.batch_size)
            
            # Unpack batch
            states = torch.cat([x[0] for x in batch])
            actions = torch.tensor([x[1] for x in batch], dtype=torch.long).to(device)
            old_action_probs = torch.cat([x[2] for x in batch])
            rewards = torch.FloatTensor([x[3] for x in batch]).to(device)
            next_states = torch.cat([x[4] for x in batch])
            dones = torch.FloatTensor([x[5] for x in batch]).to(device)
            
            # Compute returns and advantages
            with torch.no_grad():
                values = self.critic(states).squeeze()
                next_values = self.critic(next_states).squeeze()
                
                # Calculate returns with TD lambda
                returns = rewards + self.gamma * next_values * (1 - dones)
                advantages = returns - values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get the indices of the sampled actions
            action_indices = actions
            
            # Optimize policy for K epochs
            for _ in range(self.K_epochs):
                # Get current action logits and value estimates
                current_action_logits = self.actor(states)
                current_action_probs = torch.softmax(current_action_logits, dim=-1)
                current_values = self.critic(states).squeeze()
                
                # Create categorical distribution
                dist = torch.distributions.Categorical(current_action_probs)
                
                # Get log probabilities of actions
                action_log_probs = dist.log_prob(action_indices)
                dist_entropy = dist.entropy()
                
                # Get log probabilities of actions from old policy
                old_dist = torch.distributions.Categorical(old_action_probs)
                old_action_log_probs = old_dist.log_prob(action_indices)
                
                # Calculate ratio (pi_theta / pi_theta_old)
                ratios = torch.exp(action_log_probs - old_action_log_probs.detach())
                
                # Calculate surrogate losses
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
                # PPO loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Add entropy term for exploration
                actor_loss = actor_loss - 0.01 * dist_entropy.mean()
                
                # Value loss
                critic_loss = self.MseLoss(current_values, returns)
                
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
        
        # Copy new weights to old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
    
    def save(self, filename):
        """Save policy weights"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_old': self.actor_old.state_dict()
        }, filename)
    
    def load(self, filename):
        """Load policy weights"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_old.load_state_dict(checkpoint['actor_old'])