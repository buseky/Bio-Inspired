from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from RL.rl_environment import InvertedPendulumEnv

# LQR Gains tunable
LQR_K = np.array([-10.0, -19.56, -147.48, -50.26])  # [x, x_dot, theta, theta_dot]

def lqr_controller(obs):
    # obs: [x, x_dot, theta, theta_dot] (note: theta is wrapped, but near 0 it's fine)
    action = -np.dot(LQR_K, obs)
    action = np.clip(action / 10.0, -1.0, 1.0)  # Scale to action space [-1, 1] (max_force=10)
    return np.array([action])

def evaluate_lqr(num_episodes=100, render=False):
    env = InvertedPendulumEnv(render_mode="human" if render else None)
    cum_rewards = []
    success_rates = []  # 1 if didn't fall, 0 otherwise
    avg_forces = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        cum_reward = 0.0
        forces = []
        fell = False
        steps = 0
        
        while not done and steps < env.max_steps:
            action = lqr_controller(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            cum_reward += reward
            forces.append(abs(action[0] * env.max_force))  # Absolute force
            steps += 1
            
            if terminated:
                fell = True
                done = True
            elif truncated:
                done = True
            
            if render:
                env.render()
        
        cum_rewards.append(cum_reward)
        success_rates.append(1 if not fell else 0)
        avg_forces.append(np.mean(forces) if forces else 0)
        if render:
            print(f"Episode {ep+1}: Cum. Reward = {cum_reward:.2f}, Success = {not fell}, Avg Force = {np.mean(forces):.2f}")
    
    env.close()
    return {
        "avg_cum_reward": np.mean(cum_rewards),
        "std_cum_reward": np.std(cum_rewards),
        "success_rate": np.mean(success_rates),
        "avg_force": np.mean(avg_forces)
    }

# Run evaluation
results = evaluate_lqr(num_episodes=100, render=False)
print("LQR Evaluation Results:")
print(f"Average Cumulative Reward: {results['avg_cum_reward']:.2f} ± {results['std_cum_reward']:.2f}")
print(f"Success Rate: {results['success_rate']*100:.2f}%")
print(f"Average Force Magnitude: {results['avg_force']:.2f} N")

# Optional: Render one episode for visualization
evaluate_lqr(num_episodes=1, render=True)