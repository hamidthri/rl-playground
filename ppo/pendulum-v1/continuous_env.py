import gym
import numpy as np

class ContinuousEnvWrapper:
    """Wrapper for continuous action environments"""
    def __init__(self, env_name):
        """
        Initialize the continuous environment wrapper
        
        Args:
            env_name: Name of the gym environment (e.g., 'Pendulum-v1', 'LunarLanderContinuous-v2')
        """
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        
        # Extract action bounds
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
            action: Action in normalized space [-1, 1]
            
        Returns:
            next_state, reward, done, info
        """
        # Denormalize action from [-1, 1] to [low, high]
        denormalized_action = action * self.action_scale + self.action_offset
        
        # Clip action to ensure it's within bounds
        denormalized_action = np.clip(denormalized_action, self.action_low, self.action_high)
        
        # Step the environment
        next_state, reward, done, info = self.env.step(denormalized_action)
        
        return next_state, reward, done, info
    
    def close(self):
        """Close the environment"""
        self.env.close()
    
    def render(self, mode='human'):
        """Render the environment"""
        return self.env.render(mode=mode)