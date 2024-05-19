import argparse
import numpy as np
from controllers import PIDController
from tinyphysics import TinyPhysicsSimulator, TinyPhysicsModel
import concurrent.futures
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

def evaluate_single_simulation(kp, ki, kd, model, data_path, debug):
    pid_controller = PIDController(kp, ki, kd)
    simulator = TinyPhysicsSimulator(model, f"./{data_path}", controller=pid_controller, debug=debug)
    costs = simulator.rollout()
    return costs['total_cost']

def evaluate_pid(kp, ki, kd, model, data_files, debug=False):
    total_cost = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = executor.map(lambda path: evaluate_single_simulation(kp, ki, kd, model, path, debug), data_files)
        total_cost = sum(results)
    average_cost = total_cost / len(data_files)
    return average_cost

def run_generation(data_path, segment_size, population, generations, debug=False, pop_size=20, mut_rate=0.1, cross_rate=0.5):
    pop_size, generations, mut_rate, cross_rate = 20, 10, 0.1, 0.5
    parameter_bounds = {'kp': (0, 2), 'ki': (0, 0.2), 'kd': (0, 0.2)}
    
    
    data_path = Path(data_path)
    
    if data_path.is_dir():
        files = random.sample(sorted(data_path.iterdir())[100:], segment_size)
    else: 
        files = [data_path]

    # Initialize population
    population = np.random.rand(pop_size, 3)
    population[:, 0] = population[:, 0] * (parameter_bounds['kp'][1] - parameter_bounds['kp'][0]) + parameter_bounds['kp'][0]
    population[:, 1] = population[:, 1] * (parameter_bounds['ki'][1] - parameter_bounds['ki'][0]) + parameter_bounds['ki'][0]
    population[:, 2] = population[:, 2] * (parameter_bounds['kd'][1] - parameter_bounds['kd'][0]) + parameter_bounds['kd'][0]

    model = TinyPhysicsModel("./models/tinyphysics.onnx", debug=debug)

    for generation in range(generations):
        print(f"Generation {generation + 1}/{generations}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            tasks = [(ind[0], ind[1], ind[2], model, files, debug) for ind in population]
            fitness = list(tqdm(executor.map(lambda params: evaluate_pid(*params), tasks), total=len(tasks), desc="Evaluating PID"))

        fitness = np.array(fitness)
        
        # Selection
        parents_indices = np.argsort(fitness)[:pop_size // 2]
        parents = population[parents_indices]
        
        # Crossover
        children = []
        for _ in range(pop_size - len(parents)):
            if np.random.rand() < cross_rate:
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                crossover_point = np.random.randint(1, 3)
                child = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
                children.append(child)
        
        # Mutation
        for child in children:
            for i in range(3):
                if np.random.rand() < mut_rate:
                    child[i] = np.random.rand() * (parameter_bounds[list(parameter_bounds.keys())[i]][1] -
                                                    parameter_bounds[list(parameter_bounds.keys())[i]][0]) + \
                                parameter_bounds[list(parameter_bounds.keys())[i]][0]
        
        # Re-evaluate the best individual in a multithreaded way
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = [(ind[0], ind[1], ind[2], model, files, debug) for ind in population]
            reevaluated_fitness = list(tqdm(executor.map(lambda params: evaluate_pid(*params), tasks), total=len(tasks), desc="Re-evaluating Fitness"))
            
            
            
        best_index = np.argmin(reevaluated_fitness)
        best_pid = population[best_index]
        best_pid_cost = reevaluated_fitness[best_index]
        print("Current Best PID parameters:", best_pid)
        print("Total Cost for Best PID:", best_pid_cost)
        
        # Create new population
        
        if children:
            population = np.vstack([parents, children])
        else:
            population = np.copy(parents)

    best_index = np.argmin([evaluate_pid(*ind, model, files, debug) for ind in population])
    best_pid = population[best_index]
    
    return best_pid 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_segs", type=int, default=100)
    parser.add_argument("--gens", type=int, default=40)
    parser.add_argument("--pop_size", type=int, default=20)
    
    
    
    
