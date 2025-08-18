import pygame
import math
import numpy as np
import asyncio
import platform

# Pendulum parameters
M = 1.0       # Cart mass
m = 0.1       # Pendulum mass
l = 1.0       # Pendulum length
g = 9.81      # Gravity
dt = 0.02     # Time step
b_c = 0.1     # Cart damping
b_p = 0.05    # Pendulum damping
force_mag = 10.0  # Max force

# Initial state
x = 0.0
x_dot = 0.0
theta = np.random.uniform(-0.05, 0.05)  # Start near upright (theta=0)
theta_dot = 0.0
F = 0.0

# Pygame setup
pygame.init()
width, height = 600, 400
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Window boundaries in "physics units"
x_min = -width / 2 / 50 + 0.8  # Left limit, scaled
x_max = width / 2 / 50 - 0.8   # Right limit, scaled

def step(x, x_dot, theta, theta_dot, F):
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    
    # Add damping forces
    F_total = F - b_c * x_dot
    theta_dd = (g * sin_theta + cos_theta * (-F_total - m * l * theta_dot**2 * sin_theta) / (M + m) - b_p * theta_dot / (m * l)) / \
               (l * (4/3 - m * cos_theta**2 / (M + m)))
    
    x_dd = (F_total + m * l * (theta_dot**2 * sin_theta - theta_dd * cos_theta)) / (M + m)
    
    # Integrate
    x_dot += x_dd * dt
    x += x_dot * dt
    theta_dot += theta_dd * dt
    theta += theta_dot * dt
    
    # Boundary check
    if x < x_min:
        x = x_min
        x_dot = 0
    elif x > x_max:
        x = x_max
        x_dot = 0
    
    return x, x_dot, theta, theta_dot

async def main():
    global x, x_dot, theta, theta_dot, F, running
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            F = -force_mag
        elif keys[pygame.K_RIGHT]:
            F = force_mag
        else:
            F = 0.0
        
        # Step simulation
        x, x_dot, theta, theta_dot = step(x, x_dot, theta, theta_dot, F)
        
        # Draw
        screen.fill((255, 255, 255))
        
        cart_y = height - 50
        cart_width = 80
        cart_height = 20
        cart_x = int(width / 2 + x * 50)  # Scale for visualization
        
        # Pendulum end
        pend_x = int(cart_x + l * 50 * math.sin(theta))
        pend_y = int(cart_y - l * 50 * math.cos(theta))
        
        # Draw cart
        pygame.draw.rect(screen, (0, 0, 0), (cart_x - cart_width // 2, cart_y - cart_height // 2, cart_width, cart_height))
        # Draw pendulum
        pygame.draw.line(screen, (255, 0, 0), (cart_x, cart_y), (pend_x, pend_y), 4)
        pygame.draw.circle(screen, (0, 0, 255), (pend_x, pend_y), 10)
        # Optional: Draw upright reference line
        ref_x = int(cart_x)
        ref_y = int(cart_y - l * 50)  # Upright position
        pygame.draw.line(screen, (0, 255, 0), (cart_x, cart_y), (ref_x, ref_y), 2)
        
        pygame.display.flip()
        clock.tick(50)
        await asyncio.sleep(1.0 / 50)  # Control frame rate for Pyodide

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())

pygame.quit()