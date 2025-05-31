# RL-Playground

A comprehensive reinforcement learning and imitation learning toolkit supporting multiple algorithms and environments. This project provides clean, modular implementations of popular RL algorithms including PPO and SAC, along with imitation learning methods like GAIL and MaxEnt IRL.

## Features

- **Reinforcement Learning Algorithms:**
  - Proximal Policy Optimization (PPO) - discrete and continuous environments
  - Soft Actor-Critic (SAC) - continuous control
  - Model Predictive Control (MPC) - traditional control baseline

- **Imitation Learning Methods:**
  - Generative Adversarial Imitation Learning (GAIL)
  - Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)
  - Behavioral Cloning (BC)

- **Supported Environments:**
  - CartPole-v1 (discrete)
  - Pendulum-v1 (continuous)
  - MountainCar (continuous)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hamidthri/rl-playground
cd rl-playground
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Project Structure

```
rl-playground/
├── rl/                          # Reinforcement Learning algorithms
│   ├── ppo/                     # PPO implementations
│   │   ├── cartpole/           # PPO for CartPole environment
│   │   └── pendulum-v1/        # PPO for Pendulum environment
│   ├── sac/                     # SAC implementations
│   │   ├── mountain_car/       # SAC for MountainCar
│   │   └── pendulum-v1/        # SAC for Pendulum
│   └── traditional/             # Traditional control methods
├── imitation/                   # Imitation Learning methods
│   ├── bc/                     # Behavioral Cloning
│   └── irl/                    # Inverse Reinforcement Learning
│       ├── gail.py            # GAIL implementation
│       ├── max_ent.py         # MaxEnt IRL
└── models/                     # Saved models and checkpoints
```

## Quick Start

### 1. Training PPO Agent

Train a PPO agent on CartPole (required for generating expert demonstrations):

```bash
cd rl/ppo/cartpole
python main.py
```

Train PPO on Pendulum (continuous):
```bash
cd rl/ppo/pendulum-v1
python continuous_train.py
```

### 2. Training SAC Agent

Train SAC on MountainCar:
```bash
cd rl/sac/mountain_car
python train.py
```

Train SAC on Pendulum:
```bash
cd rl/sac/pendulum-v1
python train.py
```

### 3. Imitation Learning with GAIL

**Step 1:** First train a PPO expert (see above) to generate demonstrations.

**Step 2:** Train GAIL using expert demonstrations:
```bash
cd imitation/irl/train_ppo_with_irl
python main.py
```

**Step 3:** Test the trained GAIL agent:
```bash
python test.py
```

### 4. Testing Trained Agents

Test PPO CartPole agent:
```bash
cd rl/ppo/cartpole
python test.py
```

Test SAC agents:
```bash
cd rl/sac/mountain_car
python test.py
```



## Model Saving and Loading

- Trained models are automatically saved in environment-specific `models/` directories
- PPO models: `rl/ppo/{environment}/models/`
- SAC models: `rl/sac/{environment}/models/`
- GAIL models: `imitation/irl/models/`
- Training curves and evaluation results are saved alongside models

## Headless Environment Support

For headless environments (no display), the code automatically handles rendering:
- Training runs without visualization by default
- Use `render=False` parameter when available
- Results and plots are saved as files instead of displayed

## Configuration

Each algorithm implementation includes configurable hyperparameters:
- Learning rates, batch sizes, network architectures
- Training episodes, evaluation frequency
- Environment-specific parameters

Modify the hyperparameters in the respective `main.py` or training scripts.

## Dependencies

Core dependencies include:
- PyTorch
- OpenAI Gym
- NumPy
- Matplotlib (for plotting results)

See `requirements.txt` for complete list.

## Important Notes

1. **Expert Data Generation**: GAIL requires expert demonstrations. Always train a PPO agent first before running GAIL.

2. **Environment Compatibility**: 
   - PPO supports both discrete (CartPole) and continuous (Pendulum) action spaces
   - SAC is designed for continuous control tasks

3. **Model Persistence**: All trained models and results are automatically saved with timestamps and performance metrics.

## Contributing

Feel free to extend this playground with:
- Additional RL algorithms (DQN, A3C, etc.)
- More environments
- Advanced imitation learning methods
- Hyperparameter optimization tools
- code structure improvements by making it more modular
