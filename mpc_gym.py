import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def generate_expert_demos(n_episodes=100, max_steps=500):
    env = gym.make('CartPole-v1')
    demos = []
    
    for _ in range(n_episodes):
        states, actions = [], []
        state, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Expert policy (simple heuristic for CartPole)
            # Move cart in direction of the pole's lean
            action = 0 if state[2] < 0 else 1
            
            states.append(state)
            actions.append(action)
            
            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            step += 1
        
        # Only keep episodes that performed reasonably well
        if step > 20:
            demos.append((states, actions))
    
    return demos

class BCPolicy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super(BCPolicy, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)
    

def prepare_training_data(demonstrations):
    states, actions = [], []
    
    for demo_states, demo_actions in demonstrations:
        states.extend(demo_states)
        actions.extend(demo_actions)
    
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

def train_bc_policy(states, actions, epochs=10, batch_size=64):
    # Convert to PyTorch tensors
    state_tensor = torch.FloatTensor(states)
    action_tensor = torch.LongTensor(actions)
    
    # Create dataset and dataloader
    dataset = TensorDataset(state_tensor, action_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    model = BCPolicy()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_states, batch_actions in dataloader:
            # Forward pass
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, losses

def evaluate_policy(model, n_episodes=20):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get model prediction
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                logits = model(state_tensor)
                action = torch.argmax(logits).item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        rewards.append(episode_reward)
    
    print(f"Average reward over {n_episodes} episodes: {np.mean(rewards):.2f}")
    return rewards

def main():
    # Generate expert demonstrations
    print("Generating expert demonstrations...")
    demos = generate_expert_demos()
    
    # Prepare training data
    print("Preparing training data...")
    states, actions = prepare_training_data(demos)
    
    # Train model
    print("Training behavior cloning model...")
    model, losses = train_bc_policy(states, actions)
    
    # Evaluate trained model
    print("Evaluating trained model...")
    rewards = evaluate_policy(model)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Behavior Cloning Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    # Plot evaluation rewards
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(rewards)), rewards)
    plt.title('Evaluation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.axhline(y=np.mean(rewards), color='r', linestyle='-', label='Mean Reward')
    plt.legend()
    plt.show()
    # Render policy
    print("Rendering policy...")
    render_policy(model)


def render_policy(model, n_episodes=3):
    env = gym.make('CartPole-v1', render_mode='human')  # Show real-time visuals
    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                logits = model(state_tensor)
                action = torch.argmax(logits).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep + 1} return: {total_reward}")
    env.close()


if __name__ == "__main__":
    main()