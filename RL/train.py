from stable_baselines3 import PPO
from rl_environment import InvertedPendulumEnv
from stable_baselines3.common.vec_env import DummyVecEnv


# Wrap your environment in a DummyVecEnv for SB3
env = DummyVecEnv([lambda: InvertedPendulumEnv()])

# Create PPO model with tuned hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,  # Lower for smoother learning
    n_steps=2048,          # Larger batch for stability
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
)

# Train the model
model.learn(total_timesteps=100_000)

model.save("pendulum_ppo")

env.close()




