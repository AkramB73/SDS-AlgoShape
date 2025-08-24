# sds_library/diffusion.py

import numpy as np
import random
from typing import List, Tuple
from .agent import Agent
# Import the core JIT evaluator, not just the wrapper
from .evaluator import parallel_block_test_jit, full_fitness 
from .utils import sample_pixel_blocks
from .palette import Palette
from .shapes import Shape
from numba import njit, prange # Import Numba decorators
import copy

# ▼▼▼ NEW HIGH-PERFORMANCE JIT FUNCTION FOR EVALUATING ALL AGENTS ▼▼▼
@njit(parallel=True, fastmath=True)
def evaluate_population_jit(population_params, population_colors, target_image, blocks, block_size):
    """
    Evaluates the entire population in parallel using Numba.
    Each agent's evaluation is assigned to a different thread.
    """
    n_agents = population_params.shape[0]
    all_scores = np.zeros(n_agents, dtype=np.float64)

    # Numba's prange will parallelize this loop over the agents
    for i in prange(n_agents):
        agent_params = population_params[i]
        agent_colors = population_colors[i]
        
        # Call the existing block evaluator for this single agent.
        # This is a highly efficient call between two JIT-compiled functions.
        score = parallel_block_test_jit(blocks, block_size, agent_params, agent_colors, target_image)
        
        # Normalize the score
        num_pixels_checked = blocks.shape[0] * block_size * block_size
        all_scores[i] = score / num_pixels_checked if num_pixels_checked > 0 else 0.0
        
    return all_scores

class DiffusionSearch:
    def __init__(self, target, palette, n_agents=50, shapes_per_agent=25, n_samples=50, **config):
        self.target = target
        self.palette = palette
        self.n_agents = n_agents
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples
        self.config = config
        self.background_color = self.config.get('background_color', (255, 255, 255, 255))
        self.img_size = (target.shape[1], target.shape[0])
        self.population = [Agent(self.img_size, self.palette, self.shapes_per_agent) for _ in range(n_agents)]
        # The executor is no longer needed and has been removed.

    # The __del__ method is no longer needed.

    def step(self, current_iteration, total_iterations, block_size: int):
        """Performs one step of the diffusion search using the optimized Numba evaluator."""
        blocks_list = sample_pixel_blocks(self.img_size[0], self.img_size[1], self.n_samples, block_size)
        blocks_arr = np.array(blocks_list, dtype=np.int32)
        
        # Prepare all agent data into 3D NumPy arrays that Numba can process
        pop_params = np.array([agent.shape_params for agent in self.population], dtype=np.float32)
        pop_colors = np.array([agent.shape_colors for agent in self.population], dtype=np.float32)

        # A SINGLE call to the new parallel function to evaluate all agents
        all_scores = evaluate_population_jit(pop_params, pop_colors, self.target, blocks_arr, block_size)
        
        population_average_error = np.mean(all_scores)
        current_threshold = population_average_error 
        
        active_agents = [self.population[i] for i, score in enumerate(all_scores) if score < current_threshold]
        
        for agent in self.population:
            if agent not in active_agents and active_agents:
                mentor = random.choice(active_agents)
                # Use deepcopy to ensure shapes are independent
                agent.shapes = copy.deepcopy(mentor.shapes)
                agent.mutate()
            elif not active_agents:
                agent._create_random_shapes()
            
            # If the agent wasn't changed, its Numba data is still valid.
            # If it was changed, mutate() or _create_random_shapes() already updated it.

        return len(active_agents), current_threshold
        
    def run(self, iterations, block_size):
        print(f"Starting SDS with {self.n_agents} agents for {iterations} iterations...")
        print(f"Using sample block size: {block_size}x{block_size}")
        for i in range(iterations):
            active_count, thresh = self.step(current_iteration=i, total_iterations=iterations, block_size=block_size)
            print(f"Completed iteration {i+1}, active agents: {active_count}, threshold: {thresh:.2f}")

        print("SDS run complete.")

    def get_best_agents(self, n: int = 1) -> List[Agent]:
        print(f"Evaluating final population to find the top {n} agent(s)...")
        # For the final evaluation, we can just use a simple loop.
        agent_scores = []
        for agent in self.population:
            error = full_fitness(agent.shapes, self.target, self.background_color)
            agent_scores.append((error, agent))
        
        if not agent_scores:
            return []
        
        agent_scores.sort(key=lambda item: item[0])
        sorted_agents = [agent for score, agent in agent_scores]
        if sorted_agents:
            print(f"Found best agent with fitness score (RMSE): {agent_scores[0][0]:.2f}")
        
        return sorted_agents[:n]