import gym
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CartPoleEnv:
    def __init__(self):
        """
        Initialize the Gym Cart-Pole environment wrapper
        """
        self.env = gym.make('CartPole-v1')
        self.state_dim = self.env.observation_space.shape[0]  # 4 for CartPole
        self.action_dim = 2  # Discrete actions: 0 (left) or 1 (right)
        
    def reset(self):
        """Reset the environment"""
        return self.env.reset()
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: Discrete action (0 or 1)
            
        Returns:
            next_state, reward, done, info
        """
        # Take step in environment with the discrete action
        next_state, reward, done, info = self.env.step(action)
        
        return next_state, reward, done, info
    
    def close(self):
        """Close the environment"""
        self.env.close()

class CartPoleVisualizer:
    def __init__(self, env, controller=None):
        """
        Initialize visualizer for gym cart-pole system
        
        Args:
            env: CartPoleEnv environment wrapper
            controller: Controller object (optional)
        """
        self.env = env
        self.controller = controller
        self.state = self.env.reset()
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-0.5, 2.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title('CartPole with PPO Control (Gym)')
        self.ax.set_xlabel('Position (m)')
        
        # Create cart and pole objects
        self.cart_width = 0.4
        self.cart_height = 0.2
        self.pole_length = 0.5  # Half the actual pole length for visualization
        
        # Cart (rectangle)
        self.cart = plt.Rectangle(
            (0 - self.cart_width/2, 0 - self.cart_height/2),
            self.cart_width,
            self.cart_height,
            fill=True,
            color='blue',
            ec='black'
        )
        self.ax.add_patch(self.cart)
        
        # Pole (line)
        self.pole, = self.ax.plot([0, 0], [0, 0], 'k-', lw=3)
        
        # Pole tip (circle)
        self.pole_tip = plt.Circle(
            (0, 0),
            0.05,
            fill=True,
            color='red'
        )
        self.ax.add_patch(self.pole_tip)
        
        # Text for state information
        self.state_text = self.ax.text(-2.3, 2.2, '', fontsize=10)
        self.control_text = self.ax.text(-2.3, 2.0, '', fontsize=10)
        self.reward_text = self.ax.text(-2.3, 1.8, '', fontsize=10)
        
        # Store state history for plotting
        self.time_history = []
        self.state_history = []
        self.control_history = []
        self.reward_history = []
        
        # Episode tracking
        self.total_reward = 0
        self.step_count = 0
    
    def update(self, frame):
        """Update function for animation"""
        # Apply control if controller is provided
        if self.controller:
            # Use PPO policy to select action
            if hasattr(self.controller, 'select_action'):
                action, _, _ = self.controller.select_action(self.state)
            else:
                # Default behavior if select_action doesn't exist
                action = 0
                
            self.control_history.append(action)
        else:
            action = self.env.env.action_space.sample()
            self.control_history.append(action)
        
        # Step the environment
        next_state, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        self.reward_history.append(reward)
        
        # Store state for plotting
        current_time = len(self.time_history) * 0.02  # Assuming 50ms per frame
        self.time_history.append(current_time)
        self.state_history.append(next_state.copy())
        
        # Update cart position
        cart_x = next_state[0]
        self.cart.set_xy((cart_x - self.cart_width/2, -self.cart_height/2))
        
        # Convert angle from gym's format (angle from vertical) to our format
        angle = next_state[2]
        
        # Update pole
        pole_x = [cart_x, cart_x + self.pole_length * np.sin(angle)]
        pole_y = [0, self.pole_length * np.cos(angle)]
        self.pole.set_data(pole_x, pole_y)
        
        # Update pole tip
        self.pole_tip.center = (pole_x[1], pole_y[1])
        
        # Update text
        state_str = f'x: {next_state[0]:.2f} m, θ: {(next_state[2] * 180/np.pi):.1f}°\n'
        state_str += f'v: {next_state[1]:.2f} m/s, ω: {(next_state[3] * 180/np.pi):.1f}°/s'
        self.state_text.set_text(state_str)
        
        self.control_text.set_text(f'Action: {action}')
        self.reward_text.set_text(f'Total Reward: {self.total_reward}, Steps: {self.step_count}')
        
        # Reset if done
        if done:
            self.state = self.env.reset()
            self.total_reward = 0
            self.step_count = 0
            print(f"Episode ended. Steps: {self.step_count}")
        else:
            self.state = next_state
        
        return self.cart, self.pole, self.pole_tip, self.state_text, self.control_text, self.reward_text
    
    def animate(self, frames=500, interval=50):
        """Run animation"""
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval,
            blit=True
        )
        plt.show()