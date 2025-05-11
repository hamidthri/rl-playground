import gymnasium as gym
import numpy as np

class PendulumEnvWrapper:
    """
    A wrapper for the Pendulum-v1 environment with some utility functions.
    """
    def __init__(self, render_mode=None):
        self.env = gym.make('Pendulum-v1', render_mode=render_mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Action space parameters
        self.action_dim = self.action_space.shape[0]
        self.max_action = self.action_space.high[0]
        self.min_action = self.action_space.low[0]
        
        # State space parameters
        self.state_dim = self.observation_space.shape[0]
    
    def reset(self):
        return self.env.reset()[0]  # gym returns (obs, info), we just want obs
    
    def step(self, action):
        # Scale action to the proper range if needed
        scaled_action = action  # SAC produces actions in the correct range
        
        next_state, reward, terminated, truncated, info = self.env.step(scaled_action)
        done = terminated or truncated
        
        return next_state, reward, done, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()
    
    def sample_action(self):
        return self.env.action_space.sample()