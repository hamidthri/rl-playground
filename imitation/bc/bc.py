import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from ppo.cartpole.actor import ActorNetwork
import argparse
import torch.nn.functional as F


# ==== 1. Define PPO and BC Model Structures ====


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return F.relu(out + identity)

class BCPolicy(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.output_layer(x)


# ==== 2. Load PPO Model ====
def load_ppo_model(state_dim, action_dim, path="/Users/htaheri/Documents/GitHub/rl-playground/ppo/cartpole/models/ppo_CartPole_v1_final.pth"):
    checkpoint = torch.load(path)
    model = ActorNetwork(state_dim, action_dim, hidden_dim=64)
    model.load_state_dict(checkpoint["actor"])
    model.eval()
    return model



# ==== 3. Generate Expert Demonstrations ====

def collect_expert_demos(model, env_name="CartPole-v1", n_episodes=100, max_steps=500):
    env = gym.make(env_name)
    demos = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_states, ep_actions = [], []
        for _ in range(max_steps):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                logits = model(obs_tensor)
                action = logits.argmax(dim=-1).item() # ex. ligits = [0.1, 0.9], action = 1 or lo
            ep_states.append(obs)
            ep_actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        if len(ep_states) > 50:
            demos.append((ep_states, ep_actions))
    print(f"Collected {len(demos)} expert trajectories.")
    return demos


# ==== 4. Prepare Training Data ====

def prepare_training_data(demos):
    states, actions = [], []
    for s, a in demos:
        states.extend(s)
        actions.extend(a)
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)

# ==== 5. Train BC Model ====

def train_bc_policy(states, actions, state_dim, action_dim, epochs=10, batch_size=64):
    dataset = TensorDataset(torch.tensor(states), torch.tensor(actions))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BCPolicy(input_dim=state_dim, output_dim=action_dim)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch_states, batch_actions in dataloader:
            logits = model(batch_states)
            loss = criterion(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(dataloader)
        losses.append(avg)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg:.4f}")

    return model, losses


# ==== 6. Evaluate Model ====

def evaluate_policy(model, n_episodes=10):
    env = gym.make("CartPole-v1", render_mode=None)
    scores = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0
        done = False
        while not done:
            with torch.no_grad():
                logits = model(torch.tensor(obs).float())
                action = logits.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        scores.append(total)
    print(f"Average Reward: {np.mean(scores):.2f}")
    return scores


# ==== 8. Main Pipeline ====


def run_dagger_iteration(env, bc_model, expert_model, max_steps=500):
    """Run 1 DAgger iteration: collect states from BC, label with expert."""
    states = []
    expert_actions = []

    obs, _ = env.reset()
    for _ in range(max_steps):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs).float().unsqueeze(0)
            # BC picks an action
            bc_logits = bc_model(obs_tensor)
            bc_action = bc_logits.argmax(dim=-1).item()

            # Expert gives the "correct" action
            expert_logits = expert_model(obs_tensor)
            expert_action = expert_logits.argmax(dim=-1).item()

        states.append(obs)
        expert_actions.append(expert_action)

        obs, _, terminated, truncated, _ = env.step(bc_action)
        if terminated or truncated:
            break

    return np.array(states, dtype=np.float32), np.array(expert_actions, dtype=np.int64)


def render_policy(model, n_episodes=3):
    env = gym.make("CartPole-v1", render_mode="human")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0
        while not done:
            with torch.no_grad():
                logits = model(torch.tensor(obs).float())
                action = logits.argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        print(f"Episode {ep+1} return: {total}")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning + DAgger Trainer")
    parser.add_argument("--env", type=str, default="CartPole-v1", help="Gym environment")
    parser.add_argument("--dagger", action="store_true", help="Use DAgger")
    parser.add_argument("--dagger-iters", type=int, default=5, help="Number of DAgger iterations")
    parser.add_argument("--ppo-path", type=str, default="/Users/htaheri/Documents/GitHub/rl-playground/ppo/cartpole/models/ppo_cartpole_best.pth", help="Path to PPO expert model")
    args = parser.parse_args()

    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print("Loading PPO expert...")
    ppo_model = load_ppo_model(state_dim, action_dim, path=args.ppo_path)

    print("Collecting initial expert demonstrations...")
    demos = collect_expert_demos(ppo_model, env_name=args.env, n_episodes=20)
    states, actions = prepare_training_data(demos)

    print("Training initial BC policy...")
    bc_model, losses = train_bc_policy(states, actions, state_dim, action_dim)

    if args.dagger:
        dagger_env = gym.make(args.env)
        for it in range(args.dagger_iters):
            print(f"\nDAgger iteration {it + 1}...")
            new_states, new_actions = run_dagger_iteration(
                dagger_env, bc_model, ppo_model
            )
            states = np.concatenate([states, new_states], axis=0)
            actions = np.concatenate([actions, new_actions], axis=0)
            bc_model, losses = train_bc_policy(states, actions, state_dim, action_dim)
            print("Evaluating policy...")
            evaluate_policy(bc_model)

    print("Rendering final BC policy...")
    render_policy(bc_model)

    plt.plot(losses)
    plt.title("Final BC Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()




if __name__ == "__main__":
    main()
