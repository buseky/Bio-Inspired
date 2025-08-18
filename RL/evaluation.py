import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_environment import InvertedPendulumEnv

def evaluate_ppo(num_episodes=100, render=False, render_one_episode=True):
    # Load environment and model
    env = DummyVecEnv([lambda: InvertedPendulumEnv(render_mode="human" if render or render_one_episode else None)])
    model = PPO.load("pendulum_ppo", env=env)
    
    cum_rewards = []
    success_rates = []  # 1 if didn't fall, 0 otherwise
    avg_forces = []
    # Store one episode's trajectory for plotting
    if render_one_episode:
        traj_episode = num_episodes - 1  # Plot and render last episode
        states = []
        forces = []
    else:
        traj_episode = -1
        states = None
        forces = None
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = [False]
        cum_reward = 0.0
        episode_forces = []
        fell = False
        steps = 0
        
        if ep == traj_episode and render_one_episode:
            states.append(obs[0])
        
        while not done[0] and steps < env.envs[0].max_steps:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            cum_reward += reward[0]
            episode_forces.append(abs(action[0, 0] * env.envs[0].max_force))
            steps += 1
            
            if ep == traj_episode and render_one_episode:
                states.append(obs[0])
                forces.append(action[0, 0] * env.envs[0].max_force)
            
            if done[0]:  # terminated or truncated
                fell = info[0].get('terminal_observation') is not None and abs(info[0]['terminal_observation'][2]) > np.pi / 18
                done = [True]
            
            if render or (ep == traj_episode and render_one_episode):
                env.render()
                import pygame
                pygame.time.wait(1000 // 30)  # ~30 FPS for visibility
        
        cum_rewards.append(cum_reward)
        success_rates.append(1 if not fell else 0)
        avg_forces.append(np.mean(episode_forces) if episode_forces else 0)
        print(f"Episode {ep+1}: Cum. Reward = {cum_reward:.2f}, Success = {not fell}, Avg Force = {np.mean(episode_forces):.2f}")
    
    env.close()
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Histogram of cumulative rewards
    plt.subplot(2, 1, 1)
    plt.hist(cum_rewards, bins=20, color='blue', alpha=0.7)
    plt.title("Histogram of Cumulative Rewards (PPO)")
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Frequency")
    
    # State and force trajectories for one episode
    if render_one_episode and states:
        plt.subplot(2, 1, 2)
        times = np.arange(len(states)) * env.envs[0].dt
        states = np.array(states)
        plt.plot(times, states[:, 0], label="x (m)")
        plt.plot(times, states[:, 1], label="x_dot (m/s)")
        plt.plot(times, states[:, 2], label="theta (rad)")
        plt.plot(times, states[:, 3], label="theta_dot (rad/s)")
        plt.plot(times[:-1], forces, label="Force (N)", linestyle="--")
        plt.title("State and Force Trajectories (Last Episode)")
        plt.xlabel("Time (s)")
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        "avg_cum_reward": np.mean(cum_rewards),
        "std_cum_reward": np.std(cum_rewards),
        "success_rate": np.mean(success_rates),
        "avg_force": np.mean(avg_forces)
    }

# Run evaluation
results = evaluate_ppo(num_episodes=100, render=False, render_one_episode=True)
print("\nPPO Evaluation Results:")
print(f"Average Cumulative Reward: {results['avg_cum_reward']:.2f} ± {results['std_cum_reward']:.2f}")
print(f"Success Rate: {results['success_rate']*100:.2f}%")
print(f"Average Force Magnitude: {results['avg_force']:.2f} N")