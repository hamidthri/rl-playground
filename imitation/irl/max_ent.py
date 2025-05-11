import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from ppo.cartpole.actor import ActorNetwork
import argparse

class Trajectory:
    """A simple trajectory class to store state-action trajectories."""
    def __init__(self, states, actions):
        self.state_action_pairs = list(zip(states, actions))
    
    def states(self):
        """Return states from the trajectory."""
        return [s for s, _ in self.state_action_pairs]
    
    def actions(self):
        """Return actions from the trajectory."""
        return [a for _, a in self.state_action_pairs]
    
    def transitions(self):
        """Return transitions (s, a, s') from the trajectory."""
        return [(s, a, self.state_action_pairs[i+1][0]) 
                if i < len(self.state_action_pairs) - 1 
                else (s, a, None) 
                for i, (s, a) in enumerate(self.state_action_pairs)]


class FeatureExtractor:
    """Extract features from cartpole state for MaxEnt IRL."""
    def __init__(self, num_features=16):
        self.num_features = num_features
        # Create feature grid bounds based on CartPole state space
        self.position_bounds = (-2.4, 2.4)
        self.velocity_bounds = (-4.0, 4.0)
        self.angle_bounds = (-0.25, 0.25)
        self.angular_velocity_bounds = (-4.0, 4.0)
        
        # Create feature grid
        self.grid_positions = np.linspace(self.position_bounds[0], self.position_bounds[1], int(np.sqrt(num_features)))
        self.grid_velocities = np.linspace(self.velocity_bounds[0], self.velocity_bounds[1], int(np.sqrt(num_features)))
        self.grid_angles = np.linspace(self.angle_bounds[0], self.angle_bounds[1], int(np.sqrt(num_features)))
        self.grid_angular_velocities = np.linspace(self.angular_velocity_bounds[0], self.angular_velocity_bounds[1], int(np.sqrt(num_features)))
        
    def get_features(self, state):
        """Convert cartpole state to feature vector using RBF."""
        position, velocity, angle, angular_velocity = state
        
        # Compute RBF features
        features = np.zeros(self.num_features)
        
        # Create grid indices
        grid_size = int(np.sqrt(self.num_features))
        centers = list(product(
            self.grid_positions[:grid_size], 
            self.grid_velocities[:grid_size], 
            self.grid_angles[:grid_size], 
            self.grid_angular_velocities[:grid_size]
        ))[:self.num_features]

        for i, (pos_center, vel_center, ang_center, ang_vel_center) in enumerate(centers):            
            # Compute RBF for each dimension
            pos_rbf = np.exp(-0.5 * ((position - pos_center) / 0.5) ** 2)
            vel_rbf = np.exp(-0.5 * ((velocity - vel_center) / 0.5) ** 2)
            ang_rbf = np.exp(-0.5 * ((angle - ang_center) / 0.05) ** 2)
            ang_vel_rbf = np.exp(-0.5 * ((angular_velocity - ang_vel_center) / 0.5) ** 2)
            
            # Combined RBF feature
            features[i] = pos_rbf * vel_rbf * ang_rbf * ang_vel_rbf
            
        return features
    
    def get_feature_matrix(self, states):
        """Convert a list of states to a feature matrix."""
        n_states = len(states)
        features = np.zeros((n_states, self.num_features))
        for i, state in enumerate(states):
            features[i] = self.get_features(state)
        return features


class Optimizer:
    """Simple gradient ascent optimizer."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.theta = None
    
    def reset(self, theta):
        """Reset the optimizer with initial parameters."""
        self.theta = theta.copy()
    
    def step(self, gradient):
        """Update parameters based on gradient."""
        self.theta += self.learning_rate * gradient
        return self.theta


class Initializer:
    """Parameter initializer."""
    def __init__(self, method='zeros'):
        self.method = method
    
    def __call__(self, n_features):
        """Initialize parameters."""
        if self.method == 'zeros':
            return np.zeros(n_features)
        elif self.method == 'random':
            return np.random.randn(n_features) * 0.01
        else:
            raise ValueError(f"Unknown initialization method: {self.method}")


def discretize_state(state, n_bins=10):
    """Discretize continuous state for transition matrix building."""
    position, velocity, angle, angular_velocity = state
    
    # Discretize each dimension
    pos_bins = np.linspace(-2.4, 2.4, n_bins)
    vel_bins = np.linspace(-4.0, 4.0, n_bins)
    ang_bins = np.linspace(-0.25, 0.25, n_bins)
    ang_vel_bins = np.linspace(-4.0, 4.0, n_bins)
    
    pos_idx = min(n_bins - 1, max(0, np.digitize(position, pos_bins) - 1))
    vel_idx = min(n_bins - 1, max(0, np.digitize(velocity, vel_bins) - 1))
    ang_idx = min(n_bins - 1, max(0, np.digitize(angle, ang_bins) - 1))
    ang_vel_idx = min(n_bins - 1, max(0, np.digitize(angular_velocity, ang_vel_bins) - 1))
    
    # Compute flat index
    flat_idx = pos_idx * (n_bins**3) + vel_idx * (n_bins**2) + ang_idx * n_bins + ang_vel_idx
    return int(flat_idx)


def build_transition_matrix(demos, n_states, n_actions):
    """Build transition matrix from demonstrations."""
    # Initialize transition counts
    counts = np.zeros((n_states, n_states, n_actions))
    
    # Count transitions
    for trajectory in demos:
        for (s, a, s_next) in trajectory.transitions():
            if s_next is not None:  # Skip last transition
                s_idx = discretize_state(s)
                s_next_idx = discretize_state(s_next)
                counts[s_idx, s_next_idx, a] += 1
    
    # Normalize to get probabilities
    p_transition = np.zeros_like(counts)
    for s in range(n_states):
        for a in range(n_actions):
            total = counts[s, :, a].sum()
            if total > 0:
                p_transition[s, :, a] = counts[s, :, a] / total
    
    return p_transition


def collect_trajectories_from_demos(states, actions):
    """Convert states and actions to Trajectory objects."""
    # Find episode boundaries (where states reset)
    episode_ends = []
    for i in range(1, len(states)):
        # Detect large position changes (indicating reset)
        if abs(states[i][0] - states[i-1][0]) > 1.0:
            episode_ends.append(i)
    
    # Include the end of the dataset
    episode_ends.append(len(states))
    
    # Create trajectories
    trajectories = []
    start_idx = 0
    for end_idx in episode_ends:
        if end_idx - start_idx > 1:  # Only include non-trivial trajectories
            trajectory = Trajectory(states[start_idx:end_idx], actions[start_idx:end_idx])
            trajectories.append(trajectory)
        start_idx = end_idx
    
    return trajectories


def load_ppo_model(state_dim, action_dim, path):
    """Load a pre-trained PPO model."""
    checkpoint = torch.load(path)
    model = ActorNetwork(state_dim, action_dim, hidden_dim=64)
    model.load_state_dict(checkpoint["actor"])
    model.eval()
    return model


def collect_expert_demos(model, env_name="CartPole-v1", n_episodes=20, max_steps=500):
    """Collect expert demonstrations using a pre-trained model."""
    env = gym.make(env_name)
    trajectories = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        states, actions = [], []
        
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
            trajectories.append(Trajectory(states, actions))
    
    print(f"Collected {len(trajectories)} expert trajectories.")
    return trajectories


def feature_expectation_from_trajectories(features, trajectories):
    """
    Compute the feature expectation of the given trajectories.
    Simply counts the number of visitations to each feature and
    divides by the number of trajectories.
    """
    n_features = features.shape[1]
    fe = np.zeros(n_features)
    
    for t in trajectories:
        for s in t.states():
            feature_vector = features[discretize_state(s)]
            fe += feature_vector
    
    return fe / len(trajectories)


def initial_probabilities_from_trajectories(n_states, trajectories):
    """
    Compute the probability of a state being a starting state using the
    given trajectories.
    """
    p = np.zeros(n_states)
    
    for t in trajectories:
        first_state = t.states()[0]
        first_state_idx = discretize_state(first_state)
        p[first_state_idx] += 1.0
    
    return p / len(trajectories)


def expected_svf_from_policy(p_transition, p_initial, terminal_states, p_action, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.
    """
    n_states = p_transition.shape[0]
    n_actions = p_transition.shape[2]
    
    # 'fix' transition probabilities to allow for convergence
    p_transition = np.copy(p_transition)
    for terminal in terminal_states:
        p_transition[terminal, :, :] = 0.0
    
    # Set up transition matrices for each action
    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    
    # Forward computation of state expectations
    d = np.zeros(n_states)
    
    delta = np.inf
    iterations = 0
    max_iterations = 1000
    
    while delta > eps and iterations < max_iterations:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)
        
        delta = np.max(np.abs(d_ - d))
        d = d_
        iterations += 1
    
    if iterations == max_iterations:
        print("Warning: SVF computation did not converge.")
    
    return d


def local_action_probabilities(p_transition, terminal_states, reward):
    """
    Compute the local action probabilities (policy) required for MaxEnt IRL.
    """
    n_states, _, n_actions = p_transition.shape
    
    er = np.exp(reward)
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]
    
    # Initialize at terminal states
    zs = np.zeros(n_states)
    for terminal in terminal_states:
        zs[terminal] = 1.0
    
    # Perform backward pass (fixed number of iterations)
    for _ in range(2 * n_states):
        za = np.array([er * p[a].dot(zs) for a in range(n_actions)]).T
        zs = za.sum(axis=1)
    
    # Compute local action probabilities
    # Add small epsilon to avoid division by zero
    return za / (zs[:, None] + 1e-10)


def compute_expected_svf(p_transition, p_initial, terminal_states, reward, eps=1e-5):
    """
    Compute the expected state visitation frequency for MaxEnt IRL.
    """
    p_action = local_action_probabilities(p_transition, terminal_states, reward)
    return expected_svf_from_policy(p_transition, p_initial, terminal_states, p_action, eps)


def max_ent_irl(p_transition, features, terminal_states, trajectories, optimizer, initializer, 
                eps=1e-4, eps_svf=1e-5, regularization=0.0):
    """
    Maximum Entropy Inverse Reinforcement Learning algorithm.
    """
    n_states, _, n_actions = p_transition.shape
    n_features = features.shape[1]
    
    # Compute static properties from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)
    
    # Basic gradient descent
    theta = initializer(n_features)
    delta = np.inf
    
    optimizer.reset(theta)
    iteration = 0
    max_iterations = 100
    
    # Store loss history
    losses = []
    
    while delta > eps and iteration < max_iterations:
        theta_old = theta.copy()
        
        # Compute per-state reward
        reward = features.dot(theta)
        
        # Compute the gradient
        e_svf = compute_expected_svf(p_transition, p_initial, terminal_states, reward, eps_svf)
        grad = e_features - features.T.dot(e_svf)
        
        # Add L2 regularization
        if regularization > 0:
            grad -= regularization * theta
            
        # Perform optimization step and compute delta for convergence
        theta = optimizer.step(grad)
        delta = np.max(np.abs(theta_old - theta))
        
        # Compute loss (negative log likelihood)
        loss = -np.sum(reward * e_svf) + np.sum(np.log(np.sum(np.exp(reward), axis=0)))
        if regularization > 0:
            loss += 0.5 * regularization * np.sum(theta ** 2)
        losses.append(loss)

        # ADD IT HERE:
        print(f"Iteration {iteration+1}, Loss: {loss:.4f}, Delta: {delta:.6f}")

        iteration += 1

    
    # Re-compute per-state reward
    reward = features.dot(theta)
    return reward, theta, losses


def maxent_policy_from_reward(p_transition, reward):
    """Extract policy from the learned reward function."""
    n_states, _, n_actions = p_transition.shape
    
    # Compute value function
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    
    for _ in range(100):  # Value iteration
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = reward[s] + 0.99 * np.sum(p_transition[s, :, a] * V)
            V[s] = np.max(Q[s, :])
    
    # Extract policy
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        best_a = np.argmax(Q[s, :])
        policy[s, best_a] = 1.0
    
    return policy


class MaxEntPolicy(nn.Module):
    """Neural network policy based on MaxEnt IRL reward."""
    def __init__(self, state_dim=4, hidden_dim=64, action_dim=2, feature_extractor=None, theta=None):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        
        self.feature_extractor = feature_extractor
        self.theta = theta  # MaxEnt reward parameters
        
    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)
    
    def get_reward(self, state):
        """Compute reward for a state using MaxEnt parameters."""
        if self.feature_extractor is None or self.theta is None:
            return 0.0
        features = self.feature_extractor.get_features(state)
        return np.dot(features, self.theta)


def train_maxent_policy(states, actions, theta, feature_extractor, epochs=10, batch_size=64):
    """Train a neural network to approximate the MaxEnt policy."""
    # Convert states and actions to tensors
    states_tensor = torch.FloatTensor(states)
    actions_tensor = torch.LongTensor(actions)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(states_tensor, actions_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    state_dim = states.shape[1]
    action_dim = np.max(actions) + 1
    model = MaxEntPolicy(state_dim, hidden_dim=64, action_dim=action_dim, 
                         feature_extractor=feature_extractor, theta=theta)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            # Forward pass
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, losses


def evaluate_policy(model, env_name="CartPole-v1", n_episodes=10):
    """Evaluate a policy on the environment."""
    env = gym.make(env_name)
    rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = model(obs_tensor)
                action = logits.argmax().item()
            
            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    print(f"Average reward over {n_episodes} episodes: {avg_reward:.2f}")
    return rewards


def render_policy(model, env_name="CartPole-v1", n_episodes=3):
    """Render the policy in action."""
    env = gym.make(env_name, render_mode="human")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = model(obs_tensor)
                action = logits.argmax().item()
            
            # Take action in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        print(f"Episode {ep+1} reward: {episode_reward}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Maximum Entropy IRL for CartPole")
    parser.add_argument("--ppo-path", type=str, default="/Users/htaheri/Documents/GitHub/rl-playground/ppo/cartpole/models/ppo_CartPole_v1_final.pth", 
                        help="Path to PPO expert model")
    parser.add_argument("--n-states", type=int, default=10000, 
                        help="Number of discrete states")
    parser.add_argument("--n-features", type=int, default=16, 
                        help="Number of features")
    parser.add_argument("--n-expert-demos", type=int, default=20, 
                        help="Number of expert demonstrations to collect")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym environment")
    parser.add_argument("--render", action="store_true", 
                        help="Render the trained policy")
    parser.add_argument("--regularization", type=float, default=0.01, 
                        help="L2 regularization coefficient")
    args = parser.parse_args()
    
    # Create environment and get dimensions
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Load PPO expert
    print("Loading PPO expert...")
    ppo_model = load_ppo_model(state_dim, action_dim, args.ppo_path)
    
    # Collect expert demonstrations
    print(f"Collecting {args.n_expert_demos} expert demonstrations...")
    expert_trajectories = collect_expert_demos(ppo_model, args.env, args.n_expert_demos)
    
    # Extract all states and actions from trajectories
    all_states = []
    all_actions = []
    for traj in expert_trajectories:
        all_states.extend(traj.states())
        all_actions.extend(traj.actions())
    
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.int64)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(num_features=args.n_features)
    
    # Generate feature matrix for all states
    print("Creating feature matrix...")
    discretized_states = [discretize_state(s) for s in all_states]
    unique_states = np.unique(discretized_states)
    
    # Build feature matrix for unique states
    feature_matrix = np.zeros((args.n_states, args.n_features))
    for s_idx in unique_states:
        if s_idx < args.n_states:  # Ensure we don't go out of bounds
            # Find first occurrence of this discretized state
            original_idx = discretized_states.index(s_idx)
            feature_matrix[s_idx] = feature_extractor.get_features(all_states[original_idx])
    
    # Build transition matrix
    print("Building transition matrix...")
    p_transition = build_transition_matrix(expert_trajectories, args.n_states, action_dim)
    
    # Identify terminal states (states with no outgoing transitions)
    terminal_states = []
    for s in range(args.n_states):
        if np.sum(p_transition[s, :, :]) == 0:
            terminal_states.append(s)
    
    if not terminal_states:
        # If no terminal states found, use states where episode ended
        for traj in expert_trajectories:
            last_state = traj.states()[-1]
            last_state_idx = discretize_state(last_state)
            if last_state_idx not in terminal_states:
                terminal_states.append(last_state_idx)
    
    print(f"Identified {len(terminal_states)} terminal states")
    
    # Run Maximum Entropy IRL
    print("Running Maximum Entropy IRL...")
    optimizer = Optimizer(learning_rate=0.01)
    initializer = Initializer(method='zeros')
    
    reward, theta, losses = max_ent_irl(
        p_transition=p_transition,
        features=feature_matrix,
        terminal_states=terminal_states,
        trajectories=expert_trajectories,
        optimizer=optimizer,
        initializer=initializer,
        regularization=args.regularization
    )
    
    # Train a neural network policy using the learned reward
    print("Training neural network policy...")
    maxent_policy, training_losses = train_maxent_policy(
        all_states, all_actions, theta, feature_extractor
    )
    
    # Evaluate policy
    print("Evaluating policy...")
    eval_rewards = evaluate_policy(maxent_policy, args.env)
    
    # Plot losses
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("MaxEnt IRL Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(training_losses)
    plt.title("Policy Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.tight_layout()
    plt.savefig("maxent_losses.png")
    plt.show()
    
    # Render policy if requested
    if args.render:
        print("Rendering policy...")
        render_policy(maxent_policy, args.env)
    
    print("Done!")


if __name__ == "__main__":
    main()