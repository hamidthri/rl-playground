import gymnasium as gym
import numpy as np

class EnvWrapper:
    """Wrapper for both discrete and continuous action environments"""
    def __init__(self, env_name, render=False):
        """
        Initialize the environment wrapper
        
        Args:
            env_name: Name of the gym environment (e.g., 'CartPole-v1', 'Pendulum-v1')
        """
        render_mode = "human" if render else None
        self.env = gym.make(env_name, render_mode=render_mode)


        self.state_dim = self.env.observation_space.shape[0]
        
        # Determine if action space is discrete or continuous
        self.is_discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        
        if self.is_discrete:
            self.action_dim = self.env.action_space.n  # Number of discrete actions
            self.action_space_type = "discrete"
        else:
            self.action_dim = self.env.action_space.shape[0]  # Dimension of continuous action
            self.action_space_type = "continuous"
            
            # Extract action bounds for continuous actions
            self.action_low = self.env.action_space.low
            self.action_high = self.env.action_space.high
            
            # Calculate action scale and offset for normalization
            self.action_scale = (self.action_high - self.action_low) / 2.0
            self.action_offset = (self.action_high + self.action_low) / 2.0
        
    def reset(self):
        """Reset the environment"""
        return self.env.reset()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Action (discrete or normalized continuous [-1, 1])
            
        Returns:
            next_state, reward, done, info
        """
        if not self.is_discrete:
            # For continuous actions, denormalize from [-1, 1] to [low, high]
            action = action * self.action_scale + self.action_offset
            action = np.clip(action, self.action_low, self.action_high)
        
        # Step the environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info
        
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode=mode)
    
    def get_action_space_type(self):
        """Return whether the action space is discrete or continuous"""
        return self.action_space_type
    
    def sample_action(self):
        """Sample a random action from the environment's action space"""
        if self.is_discrete:
            return self.env.action_space.sample()  # Returns an integer
        else:
            # For continuous, return normalized action in [-1, 1]
            action = self.env.action_space.sample()
            return (action - self.action_offset) / self.action_scale