import numpy as np
import random
from typing import List, Tuple
import copy

from .agent import Agent
from .evaluator import MetalEvaluator, NumbaEvaluator, full_fitness, METAL_AVAILABLE
from .utils import sample_pixel_blocks
from .palette import Palette
from .shapes import Shape, Triangle # Triangle for default value

class DiffusionSearch:
    def __init__(self, target, palette, n_agents=50, shapes_per_agent=25, n_samples=50, block_size=5, shape_types=None, **config):
        self.target = target
        self.palette = palette
        self.n_agents = n_agents
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples
        self.block_size = block_size
        self.config = config
        
        if shape_types is None:
            self.shape_types = [Triangle]
        else:
            self.shape_types = shape_types
            
        self.background_color = self.config.get('background_color', (255, 255, 255, 255))
        self.img_size = (target.shape[1], target.shape[0])
        
        self.population = [Agent(self.img_size, self.palette, self.shapes_per_agent, self.shape_types) for _ in range(n_agents)]

        if METAL_AVAILABLE:
            try:
                self.evaluator = MetalEvaluator(
                    target_image=self.target,
                    shapes_per_agent=self.shapes_per_agent,
                    n_samples=self.n_samples,
                    n_agents=self.n_agents,
                    block_size=self.block_size
                )
            except Exception as e:
                print(f"Metal initialization failed: {e}. Falling back to Numba.")
                self.evaluator = NumbaEvaluator(target_image=self.target)
        else:
            self.evaluator = NumbaEvaluator(target_image=self.target)

    def step(self, current_iteration, total_iterations):
        blocks_list = sample_pixel_blocks(self.img_size[0], self.img_size[1], self.n_samples, self.block_size)
        blocks_arr = np.array(blocks_list, dtype=np.int32)
        all_scores = self.evaluator.evaluate(self.population, blocks_arr)
        if all_scores.size == 0:
            return 0, 0.0
        population_average_error = np.mean(all_scores)
        current_threshold = population_average_error
        active_agents = [self.population[i] for i, score in enumerate(all_scores) if score < current_threshold]
        inactive_agents = [agent for agent in self.population if agent not in active_agents]
        num_to_reinitialize = int(len(inactive_agents) * 0.2)
        agents_to_reinitialize = random.sample(inactive_agents, num_to_reinitialize)
        for agent in self.population:
            if not active_agents:
                agent._create_random_shapes()
            elif agent in inactive_agents:
                if agent in agents_to_reinitialize:
                    agent._create_random_shapes()
                else:
                    mentor = random.choice(active_agents)
                    agent.shapes = copy.deepcopy(mentor.shapes)
                    agent.mutate()
        return len(active_agents), current_threshold

    def run(self, iterations):
        print(f"Starting SDS with {self.n_agents} agents for {iterations} iterations...")
        print(f"Using sample block size: {self.block_size}x{self.block_size}")
        for i in range(iterations):
            active_count, thresh = self.step(current_iteration=i, total_iterations=iterations)
            if i % 50 ==0:
                print(f"Completed iteration {i}, active agents: {active_count}, threshold: {thresh:.2f}")
        print("SDS run complete.")

    def get_best_agents(self, n: int = 1) -> List[Agent]:
        print(f"Evaluating final population to find the top {n} agent(s)...")
        agent_scores = []
        for agent in self.population:
            error = full_fitness(agent.shapes, self.target, self.background_color)
            agent_scores.append((error, agent))
        if not agent_scores: return []
        agent_scores.sort(key=lambda item: item[0])
        sorted_agents = [agent for score, agent in agent_scores]
        if sorted_agents:
            print(f"Found best agent with fitness score (RMSE): {agent_scores[0][0]:.2f}")
        return sorted_agents[:n]