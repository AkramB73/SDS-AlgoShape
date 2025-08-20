# sds_library/diffusion.py

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple
from .agent import Agent
from .evaluator import partial_test, full_fitness
from .utils import sample_pixels
from .palette import Palette
from .shapes import Shape

# --- Wrappers for Parallelism (Unchanged) ---
def _parallel_test_wrapper(args: Tuple) -> float:
    agent, target, samples = args
    return partial_test(agent, target, samples)

def _parallel_full_fitness_wrapper(args: Tuple) -> float:
    agent_shapes, target, background = args
    return full_fitness(agent_shapes, target, background)

class DiffusionSearch:
    def __init__(self, target, palette, n_agents=50, shapes_per_agent=25, n_samples=50, start_threshold=255.0, end_threshold=25.0, background_color=(255, 255, 255, 255)):
        self.target = target
        self.palette = palette
        self.n_agents = n_agents
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.background_color = background_color
        self.img_size = (target.shape[1], target.shape[0])
        self.population = [Agent(self.img_size, self.palette, self.shapes_per_agent) for _ in range(n_agents)]
        
        # --- FIX: Create the process pool ONCE during initialization ---
        self.executor = ProcessPoolExecutor()

    def __del__(self):
        """Ensure the executor is shut down when the object is destroyed."""
        if self.executor:
            self.executor.shutdown()

    def _calculate_annealing_threshold(self, current_iteration, total_iterations):
        progress = current_iteration / total_iterations
        decay_factor = np.exp(-7.0 * progress)
        return self.end_threshold + (self.start_threshold - self.end_threshold) * decay_factor

    def step(self, current_iteration, total_iterations):
        samples = sample_pixels(self.img_size[0], self.img_size[1], self.n_samples)
        
        all_scores = [0.0] * self.n_agents
        
        # --- FIX: Reuse the existing executor instead of creating a new one ---
        tasks_args = [(agent, self.target, samples) for agent in self.population]
        future_to_idx = {self.executor.submit(_parallel_test_wrapper, args): i for i, args in enumerate(tasks_args)}
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                all_scores[idx] = future.result()
            except Exception as e:
                all_scores[idx] = float('inf')

        # (The rest of the step logic is unchanged)
        population_average_error = np.mean([s for s in all_scores if s != float('inf')]) if any(s != float('inf') for s in all_scores) else float('inf')
        annealing_threshold = self._calculate_annealing_threshold(current_iteration, total_iterations)
        current_threshold = min(population_average_error, annealing_threshold)
        
        inactive_indices = []
        for i, score in enumerate(all_scores):
            self.population[i].is_active = score < current_threshold
            if not self.population[i].is_active:
                inactive_indices.append(i)

        random.shuffle(inactive_indices)
        for inactive_idx in inactive_indices:
            communicating_idx = random.choice(range(self.n_agents))
            if self.population[communicating_idx].is_active:
                self.population[inactive_idx].shapes = list(self.population[communicating_idx].shapes)
                self.population[inactive_idx]._prepare_data_for_numba() 
            else:
                self.population[inactive_idx]._create_random_shapes()

    def run(self, iterations):
        print(f"Starting SDS with {self.n_agents} agents for {iterations} iterations...")
        for i in range(iterations):
            self.step(i, iterations)
            print(f"Completed iteration {i + 1}/{iterations}")
        print("SDS run complete.")

    def get_best_agents(self, n: int = 1) -> List[Agent]:
        print(f"Evaluating final population to find the top {n} agent(s)...")
        agent_scores = []
        # We can still create a temporary executor here as it's a one-off task
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
            print("Could not evaluate any agents.")
            return []
        
        agent_scores.sort(key=lambda item: item[0])
        sorted_agents = [agent for score, agent in agent_scores]
        if sorted_agents:
            print(f"Found best agent with fitness score (RMSE): {agent_scores[0][0]:.2f}")
        
        return sorted_agents[:n]