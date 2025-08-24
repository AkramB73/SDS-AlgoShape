# sds_library/diffusion.py

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from .agent import Agent
from .evaluator import partial_test, full_fitness
# Import the new block sampling function
from .utils import sample_pixel_blocks 
from .palette import Palette
from .shapes import Shape

# --- Wrappers for Parallelism ---
# The wrapper now needs to pass the block_size to the evaluator
def _parallel_test_wrapper(args: Tuple) -> float:
    agent, target, blocks, block_size = args
    return partial_test(agent, target, blocks, block_size)

def _parallel_full_fitness_wrapper(args: Tuple) -> float:
    agent_shapes, target, background = args
    return full_fitness(agent_shapes, target, background)

class DiffusionSearch:
    def __init__(self, target, palette, n_agents=50, shapes_per_agent=25, n_samples=50, **config):
        self.target = target
        self.palette = palette
        self.n_agents = n_agents
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples
        self.config = config # Store any extra config
        self.background_color = self.config.get('background_color', (255, 255, 255, 255))
        self.img_size = (target.shape[1], target.shape[0])
        self.population = [Agent(self.img_size, self.palette, self.shapes_per_agent) for _ in range(n_agents)]
        self.executor = ProcessPoolExecutor()

    def __del__(self):
        """Ensure the executor is shut down when the object is destroyed."""
        if self.executor:
            self.executor.shutdown()

    def step(self, current_iteration, total_iterations, block_size: int):
        """Performs one step of the diffusion search using blocks."""
        # Use the new block sampler from utils.py
        blocks = sample_pixel_blocks(self.img_size[0], self.img_size[1], self.n_samples, block_size)
        
        all_scores = [0.0] * self.n_agents
        # Pass the blocks and block_size to the parallel wrapper
        tasks_args = [(agent, self.target, blocks, block_size) for agent in self.population]
        
        future_to_idx = {self.executor.submit(_parallel_test_wrapper, args): i for i, args in enumerate(tasks_args)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_scores[idx] = future.result()
            except Exception as e:
                all_scores[idx] = float('inf')
        
        population_average_error = np.mean([s for s in all_scores if s != float('inf')]) if any(s != float('inf') for s in all_scores) else float('inf')
        
        # Use a multiplier to make the threshold slightly stricter than the average
        multiplier = self.config.get('threshold_multiplier', 0.98)
        current_threshold = population_average_error * multiplier
        
        inactive_indices = []
        for i, score in enumerate(all_scores):
            self.population[i].is_active = score < current_threshold
            if not self.population[i].is_active:
                inactive_indices.append(i)

        # Learning step
        random.shuffle(inactive_indices)
        for inactive_idx in inactive_indices:
            communicating_idx = random.choice(range(self.n_agents))
            inactive_agent = self.population[inactive_idx]

            if self.population[communicating_idx].is_active:
                active_agent = self.population[communicating_idx]
                inactive_agent.shapes = list(active_agent.shapes) 
                # Call the new, more intelligent mutation method
                inactive_agent.mutate() 
            else: 
                inactive_agent._create_random_shapes()
            
        return len(inactive_indices), current_threshold
        
    def run(self, iterations, block_size):
        print(f"Starting SDS with {self.n_agents} agents for {iterations} iterations...")
        print(f"Using sample block size: {block_size}x{block_size}")
        for i in range(iterations):
            inactiveAgents, thresh = self.step(current_iteration=i, total_iterations=iterations, block_size=block_size)
            print(f"Completed iteration {i+1}, inactive agents: {inactiveAgents}, threshold: {thresh:.2f}")

        print("SDS run complete.")

    def get_best_agents(self, n: int = 1) -> List[Agent]:
        print(f"Evaluating final population to find the top {n} agent(s)...")
        agent_scores = []
        with ProcessPoolExecutor() as executor:
            tasks_args = [(agent.shapes, self.target, self.background_color) for agent in self.population]
            future_to_agent = {executor.submit(_parallel_full_fitness_wrapper, args): agent for args, agent in zip(tasks_args, self.population)}
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    error = future.result()
                    agent_scores.append((error, agent))
                except Exception as exc:
                    print(f'Agent evaluation generated an exception: {exc}')
        
        if not agent_scores:
            return []
        
        agent_scores.sort(key=lambda item: item[0])
        sorted_agents = [agent for score, agent in agent_scores]
        if sorted_agents:
            print(f"Found best agent with fitness score (RMSE): {agent_scores[0][0]:.2f}")
        
        return sorted_agents[:n]