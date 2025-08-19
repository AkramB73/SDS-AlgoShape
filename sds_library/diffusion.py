import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from .agent import Agent
from .evaluator import partial_test, full_fitness
from .utils import sample_pixels
from .palette import Palette

def _parallel_test_wrapper(args: Tuple) -> float:
    agent, target, samples = args
    return partial_test(agent, target, samples)

def _parallel_full_fitness_wrapper(args: Tuple) -> float:
    agent, target, background = args
    return full_fitness(agent, target, background)

class DiffusionSearch:
    def __init__(self,
                 target: np.ndarray,
                 palette: Palette,
                 n_agents: int = 50,
                 shapes_per_agent: int = 25,
                 n_samples: int = 50,
                 start_threshold: float = 255.0,
                 end_threshold: float = 25.0,
                 background_color: tuple = (255, 255, 255, 255)):
        
        self.target = target
        self.palette = palette
        self.n_agents = n_agents
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples
        self.start_threshold = start_threshold
        self.end_threshold = end_threshold
        self.background_color = background_color
        self.img_size = (target.shape[1], target.shape[0])
        self.population: List[Agent] = [
            Agent(self.img_size, self.palette, self.shapes_per_agent) for _ in range(n_agents)
        ]

    def _calculate_annealing_threshold(self, current_iteration: int, total_iterations: int) -> float:
        progress = current_iteration / total_iterations
        decay_factor = np.exp(-7.0 * progress)
        return self.end_threshold + (self.start_threshold - self.end_threshold) * decay_factor

    def _evaluate_population(self, samples: List[Tuple[int, int]]) -> List[float]:
        all_scores = [float('inf')] * self.n_agents
        with ProcessPoolExecutor() as executor:
            tasks_args = [(agent, self.target, samples) for agent in self.population]
            future_to_idx = {executor.submit(_parallel_test_wrapper, args): i for i, args in enumerate(tasks_args)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    score = future.result()
                    all_scores[idx] = score
                except Exception as exc:
                    print(f"Agent evaluation failed for agent {idx}: {exc}")
                    # Keep score at infinity to penalise failing agents
        return all_scores

    def step(self, current_iteration: int, total_iterations: int):

        samples = sample_pixels(self.img_size[0], self.img_size[1], self.n_samples)
        all_scores = self._evaluate_population(samples)

        population_average_error = np.mean([s for s in all_scores if s != float('inf')])
        annealing_threshold = self._calculate_annealing_threshold(current_iteration, total_iterations)
        current_threshold = min(population_average_error, annealing_threshold)
        
        inactive_indices = []
        for i, score in enumerate(all_scores):
            is_active = score < current_threshold
            self.population[i].is_active = is_active
            if not is_active:
                inactive_indices.append(i)

        random.shuffle(inactive_indices)
        for inactive_idx in inactive_indices:
            communicating_idx = random.choice(range(self.n_agents))
            
            if self.population[communicating_idx].is_active:
                self.population[inactive_idx].shapes = list(self.population[communicating_idx].shapes)
            else:
                self.population[inactive_idx]._create_random_shapes()

    def run(self, iterations: int):
        print(f"Starting SDS with {self.n_agents} agents for {iterations} iterations...")
        for i in range(iterations):
            self.step(current_iteration=i, total_iterations=iterations)
            if (i + 1) % 100 == 0:
                print(f"Completed iteration {i + 1}/{iterations}")
        print("SDS run complete.")
        
    def get_best_agents(self, n: int = 1) -> List[Agent]:
        if n > self.n_agents:
            print(f"Warning: Requested top {n} agents, but population is only {self.n_agents}. Returning all agents.")
            n = self.n_agents

        print(f"Evaluating final population to find the top {n} agent(s)")
        
        agent_scores = []
        with ProcessPoolExecutor() as executor:
            tasks_args = [(agent, self.target, self.background_color) for agent in self.population]
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
        print(f"Found best agent with fitness score (RMSE): {agent_scores[0][0]:.2f}")
        
        return sorted_agents[:n]
    