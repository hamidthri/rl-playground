import gymnasium as gym
import numpy as np

class EnvWrapper:
    """
    A wrapper for gym environments that provides a consistent interface.
    """
    def __init__(self, env_name, render=False):
        """
        Initialize the environment wrapper.
        
        Args:
            env_name: Name of the gym environment
            render: Whether to render the environment
        """
        # For newer gymnasium versions, render_mode is passed at creation time
        self.render_mode = "human" if render else None
        self.env = gym.make(env_name, render_mode=self.render_mode)
        
        # Get environment properties
        self.state_dim = self.env.observation_space.shape[0]
        
        # Handle both discrete and continuous action spaces
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
            self.action_high = self.env.action_space.high
            self.action_low = self.env.action_space.low
    
    def reset(self):
        """Reset the environment."""
        return self.env.reset()
    
    def step(self, action):
        """Take a step in the environment."""
        # Handle continuous action spaces differently if needed
        if not self.is_discrete:
            # Clip action to valid range
            action = np.clip(action, self.action_low, self.action_high)
        
        return self.env.step(action)
    
    def render(self):
        """Render the environment."""
        # In newer Gymnasium versions, render() doesn't take mode parameter
        # Rendering is handled by setting render_mode at environment creation
        if self.render_mode:
            return self.env.render()
    
    def close(self):
        """Close the environment."""
        self.env.close()