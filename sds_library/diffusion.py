# sds_library/diffusion.py

import numpy as np
import random
from typing import List, Tuple
import copy

from .agent import Agent
from .evaluator import MetalEvaluator, full_fitness
from .utils import sample_pixel_blocks
from .palette import Palette
from .shapes import Shape

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

        # --- MODIFIED: Pass n_agents to the evaluator ---
        self.evaluator = MetalEvaluator(
            target_image=self.target,
            shapes_per_agent=self.shapes_per_agent,
            n_samples=self.n_samples,
            n_agents=self.n_agents
        )

    def step(self, current_iteration, total_iterations, block_size: int):
        """Performs one step of the diffusion search using the optimized Metal evaluator."""
        blocks_list = sample_pixel_blocks(self.img_size[0], self.img_size[1], self.n_samples, block_size)
        blocks_arr = np.array(blocks_list, dtype=np.int32)

        all_scores = self.evaluator.evaluate(self.population, blocks_arr)

        if all_scores.size == 0:
            return 0, 0.0

        population_average_error = np.mean(all_scores)
        current_threshold = population_average_error

        active_agents = [self.population[i] for i, score in enumerate(all_scores) if score < current_threshold]

        for agent in self.population:
            if agent not in active_agents and active_agents:
                mentor = random.choice(active_agents)
                agent.shapes = copy.deepcopy(mentor.shapes)
                agent.mutate()
            elif not active_agents:
                agent._create_random_shapes()

        return len(active_agents), current_threshold

    def run(self, iterations, block_size):
        print(f"Starting SDS with {self.n_agents} agents for {iterations} iterations...")
        print("Using Metal for GPU acceleration.")
        print(f"Using sample block size: {block_size}x{block_size}")

        for i in range(iterations):
            active_count, thresh = self.step(current_iteration=i, total_iterations=iterations, block_size=block_size)
            print(f"Completed iteration {i+1}, active agents: {active_count}, threshold: {thresh:.2f}")

        print("SDS run complete.")

    def get_best_agents(self, n: int = 1) -> List[Agent]:
        print(f"Evaluating final population to find the top {n} agent(s)...")
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