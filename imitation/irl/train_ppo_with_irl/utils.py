import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

def plot_training_stats(reward_history, episode_lengths, show_plot=False, save_plot=False):
    """Plot the reward history and episode lengths"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(reward_history)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    
    window_size = 100
    if len(reward_history) >= window_size:
        moving_avg = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(range(window_size - 1, len(reward_history)), moving_avg, 'r-', label='Moving Average')
        ax1.legend()
    
    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Episode Lengths')
    
    if len(episode_lengths) >= window_size:
        moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(range(window_size - 1, len(episode_lengths)), moving_avg, 'r-', label='Moving Average')
        ax2.legend()
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/training_stats.png")
    
    if show_plot:
        plt.show()
    
    plt.close()  # Close figure to avoid memory issues

    

class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, action_probs, reward, next_state, done):
        self.memory.append((state, action, action_probs, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)