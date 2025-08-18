import gymnasium as gym
from gym import spaces
import numpy as np
import math
import pygame

class InvertedPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        
        # Physics
        self.M = 1.0
        self.m = 0.1
        self.l = 1.0
        self.g = 9.81
        self.dt = 0.02
        self.b_c = 0.1
        self.b_p = 0.05
        self.x_min = -5.0
        self.x_max = 5.0
        self.max_force = 10.0
        self.max_steps = 500  # Episode timeout
        
        # State: [x, x_dot, theta, theta_dot]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )
        
        # Action
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Rendering
        self.screen = None
        self.clock = None
        self.current_step = 0
        
        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = np.random.uniform(-0.05, 0.05)  # Start near upright (theta=0)
        self.theta_dot = 0.0
        self.current_step = 0
        return self._get_obs(), {}
    
    def _get_obs(self):
        # Wrap theta to [-pi, pi] for observation
        wrapped_theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        return np.array([self.x, self.x_dot, wrapped_theta, self.theta_dot], dtype=np.float32)
    
    def step(self, action):
        F = float(action[0]) * self.max_force
        
        # Equations of motion with friction
        sin_theta = math.sin(self.theta)
        cos_theta = math.cos(self.theta)
        F_total = F - self.b_c * self.x_dot
        
        theta_dd = (self.g * sin_theta + cos_theta * (-F_total - self.m * self.l * self.theta_dot**2 * sin_theta) / (self.M + self.m) - self.b_p * self.theta_dot / (self.m * self.l)) / \
                   (self.l * (4/3 - self.m * cos_theta**2 / (self.M + self.m)))
        x_dd = (F_total + self.m * self.l * (self.theta_dot**2 * sin_theta - theta_dd * cos_theta)) / (self.M + self.m)
        
        # Integrate
        self.x_dot += x_dd * self.dt
        self.x += self.x_dot * self.dt
        self.theta_dot += theta_dd * self.dt
        self.theta += self.theta_dot * self.dt
        
        # Boundary
        if self.x < self.x_min:
            self.x = self.x_min
            self.x_dot = 0
        elif self.x > self.x_max:
            self.x = self.x_max
            self.x_dot = 0
        
        # Reward: Maximize at theta=0 (upright)
        reward = np.cos(self.theta) - 0.1 * self.x**2 - 0.01 * self.x_dot**2 - 0.01 * self.theta_dot**2
        
        # Termination
        self.current_step += 1
        terminated = abs(self.theta) > math.pi / 12  # Fail if too far from upright
        truncated = self.current_step >= self.max_steps  # Time limit
        
        return self._get_obs(), reward, terminated, truncated, {}
    
    def render(self):
        if self.render_mode is None:
            raise ValueError("No render_mode was set when creating the environment.")
        
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.width, self.height = 600, 400
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            self.screen.fill((255, 255, 255))
            
            cart_y = self.height - 50
            cart_width = 80
            cart_height = 20
            cart_x = int(self.width / 2 + self.x * 50)
            
            pend_x = int(cart_x + self.l * 50 * math.sin(self.theta))
            pend_y = int(cart_y - self.l * 50 * math.cos(self.theta))
            
            pygame.draw.rect(self.screen, (0, 0, 0),
                             (cart_x - cart_width // 2, cart_y - cart_height // 2, cart_width, cart_height))
            pygame.draw.line(self.screen, (255, 0, 0), (cart_x, cart_y), (pend_x, pend_y), 4)
            pygame.draw.circle(self.screen, (0, 0, 255), (pend_x, pend_y), 10)
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None