import torch
import gymnasium as gym
from ppo.cartpole.actor import ActorNetwork

def load_policy(path, state_dim, action_dim):
    model = ActorNetwork(state_dim, action_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def test_policy(policy, env_name="CartPole-v1", render=True, episodes=10):
    env = gym.make(env_name, render_mode="human" if render else None)

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(policy(state_tensor), dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            state = next_state

        print(f"Episode {ep+1} Total Reward: {total_reward}")
    
    env.close()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    policy = load_policy("irl_learned_policy.pth", state_dim, action_dim)
    test_policy(policy)
