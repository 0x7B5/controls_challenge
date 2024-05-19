import numpy as np
import random
import concurrent.futures
from pathlib import Path
from tqdm import tqdm
import argparse
from controllers import PIDController
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel


'''
This provides two algorithms for tuning a PID controller in this context: genetic algorithm and particle swarm optimization. I have found that the genetic algorithm generally works better because PSO gets stuck on local minima often. 

Overall though, this is a pretty naive approach since it trains on a small subset of data. 
'''


'''
Run PSO with 0.00001 values for ki and kd 
'''

BOUNDS = {'kp': (0.05, 2), 'ki': (0.0001, 1), 'kd': (0.0001,1)}
class Particle:
    def __init__(self):
        self.position = np.array([random.uniform(BOUNDS['kp'][0], BOUNDS['kp'][1]), random.uniform(BOUNDS['ki'][0], BOUNDS['ki'][1]), random.uniform(BOUNDS['kd'][0], BOUNDS['kd'][1])])
        self.velocity = np.random.uniform(-1, 1, 3)
        self.p_i = self.position.copy()
        self.best_value = float('inf')

    def update_velocity(self, gbest_pos, w=0.5, c_1=1.5, c_2=1.5):
        cognitive = c_1 * random.random() * (self.p_i - self.position)
        social = c_2 * random.random() * (gbest_pos - self.position)
        self.velocity = (w * self.velocity) + cognitive + social

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, [BOUNDS['kp'][0], BOUNDS['ki'][0], BOUNDS['kd'][0]], [BOUNDS['kp'][1], BOUNDS['ki'][1], BOUNDS['kd'][1]])

def evaluate_single_simulation(kp, ki, kd, model, data_path, debug):
    pid_controller = PIDController(kp, ki, kd)
    simulator = TinyPhysicsSimulator(model, f"./{data_path}", controller=pid_controller, debug=debug)
    costs = simulator.rollout()
    return costs['total_cost']

def evaluate_pid(kp, ki, kd, model, data_files, debug=False):
    total_cost = 0
    for data_file in data_files:
        total_cost += evaluate_single_simulation(kp, ki, kd, model, data_file, debug)
    average_cost = total_cost / len(data_files)
    return average_cost

def run_pso(num_segs, iter_size, num_particles=20, iterations=40, debug=False):
    data_path = Path("./data/")
    all_files = sorted(data_path.glob("*.csv"))[:num_segs]

    swarm = [Particle() for _ in range(num_particles)]
    global_best_position = None
    global_best_value = float('inf')
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=debug)
    
    for iteration in range(iterations):
        print(f"Iteration: {iteration + 1}/{iterations}")
        
        selected_files = random.sample(all_files, iter_size)

        fitness = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [(p.position[0], p.position[1], p.position[2], model, selected_files, debug) for p in swarm]
            fitness = list(tqdm(executor.map(lambda params: evaluate_pid(*params), tasks), total=len(tasks), desc="Evaluating PID"))
        
        for i, particle in enumerate(swarm):
            if fitness[i] < particle.best_value:
                particle.best_value = fitness[i]
                particle.p_i = particle.position.copy()
                
            if fitness[i] < global_best_value:
                global_best_value = fitness[i]
                global_best_position = particle.position.copy()
        
        for particle in swarm:
            particle.update_velocity(global_best_position)
            particle.update_position()
        
        print(f"best parameters so far: {global_best_position}")
        print(f"avg total cost for these params: {global_best_value}")
    
    return global_best_position


def run_ga(num_segs, iter_size, pop_size=20, iterations=40, mut_rate=0.1, cross_rate=0.5, debug=False):
    data_path = Path("./data/")
    all_files = sorted(data_path.glob("*.csv"))[:num_segs]
    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=debug) 
    
    population = np.random.rand(pop_size, 3)
    population[:, 0] = population[:, 0] * (BOUNDS['kp'][1] - BOUNDS['kp'][0]) + BOUNDS['kp'][0]
    population[:, 1] = population[:, 1] * (BOUNDS['ki'][1] - BOUNDS['ki'][0]) + BOUNDS['ki'][0]
    population[:, 2] = population[:, 2] * (BOUNDS['kd'][1] - BOUNDS['kd'][0]) + BOUNDS['kd'][0]
    
    for generation in range(iterations): 
        print(f"Generation {generation + 1}/{iterations}")
        
        selected_files = random.sample(all_files, iter_size)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [(ind[0], ind[1], ind[2], model, selected_files, debug) for ind in population]
            fitness = list(tqdm(executor.map(lambda params: evaluate_pid(*params), tasks), total=len(tasks), desc="Evaluating PID"))

        fitness = np.array(fitness)
        parents_indices = np.argsort(fitness)[:pop_size // 2]
        parents = population[parents_indices]
    
        children = []
        for _ in range(pop_size - len(parents)):
            if np.random.rand() < cross_rate:
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                crossover_point = np.random.randint(1, 3)
                child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
                children.append(child)
        
        for child in children:
            for i in range(3):
                if np.random.rand() < mut_rate:
                    child[i] = np.random.rand() * (BOUNDS[list(BOUNDS.keys())[i]][1] - BOUNDS[list(BOUNDS.keys())[i]][0]) + BOUNDS[list(BOUNDS.keys())[i]][0]
                                
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [(ind[0], ind[1], ind[2], model, selected_files, debug) for ind in population]
            reevaluated_fitness = list(tqdm(executor.map(lambda params: evaluate_pid(*params), tasks), total=len(tasks), desc="Re-evaluating Fitness"))
            
        best_index = np.argmin(reevaluated_fitness)
        best_pid = population[best_index]
        best_pid_cost = reevaluated_fitness[best_index]
        print("best parameters so far:", best_pid)
        print("avg total cost for these params:", best_pid_cost)
        if children:
            population = np.vstack([parents, children])
        else:
            population = np.copy(parents)
    
    # selected_files = all_files
    
    best_index = np.argmin([evaluate_pid(*ind, model, selected_files, debug) for ind in population])
    best_pid = population[best_index]
    
    return best_pid 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_size", type=int, default=5)
    parser.add_argument("--num_segs", type=int, default=5)
    parser.add_argument("--algo", type=str, default="ga", choices=["ga", "pso"])
    
    
    args = parser.parse_args()
    
    if args.algo == "ga":
        best_pid = run_ga(args.num_segs, args.iter_size)
    else: 
        best_pid = run_pso(args.num_segs, args.iter_size)
    print("Best PID parameters found:", best_pid)