import numpy as np
import random


# First we define an objective function that we're trying to minimize  
def objective_function(position): 
    x, y = position
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)


# Define the Particles Used 
class Particle: 
    def __init__(self, bounds):
        self.position = np.array([random.uniform(bounds[0][0], bounds[0][1]),
                                  random.uniform(bounds[1][0], bounds[1][1])])
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        self.best_position = self.position.copy()
        self.best_value = objective_function(self.position)

# Initialize the Swarm
def initialize_swarm(num_particles, bounds):
    return [Particle(bounds) for _ in range(num_particles)]

# Define the function to update the particles
def update_particles(swarm, global_best_position, w=0.5, c1=1.5, c2=1.5):
    for particle in swarm:
        inertia = w * particle.velocity
        cognitive = c1 * random.random() * (particle.best_position - particle.position)
        social = c2 * random.random() * (global_best_position - particle.position)
        particle.velocity = inertia + cognitive + social
        particle.position += particle.velocity

def particle_swarm_optimization(objective_function, bounds, num_particles, max_iterations):
    swarm = initialize_swarm(num_particles, bounds)
    global_best_position = None
    global_best_value = float('inf')
    
    for iteration in range(max_iterations):
        for particle in swarm:
            value = objective_function(particle.position)
            if value < particle.best_value:
                particle.best_position = particle.position.copy()
                particle.best_value = value
            
            if value < global_best_value:
                global_best_position = particle.position.copy()
                global_best_value = value
        
        update_particles(swarm, global_best_position)
        print(f"Iteration {iteration + 1}/{max_iterations}, Best Value: {global_best_value}")
    
    return global_best_position, global_best_value

# Define bounds for the search space
bounds = [(-10, 10), (-10, 10)]
num_particles = 30
max_iterations = 100

# Run PSO
best_position, best_value = particle_swarm_optimization(objective_function, bounds, num_particles, max_iterations)
print(f"Best Position: {best_position}, Best Value: {best_value}")