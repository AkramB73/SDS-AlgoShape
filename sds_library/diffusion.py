# sds_library/diffusion.py

import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from .agent import Agent
from .evaluator import partial_test, full_fitness
from .utils import sample_pixels

# --- Wrapper for Parallelism (Unchanged) ---
def _parallel_test_wrapper(args: Tuple) -> float:
    agent, target_sector, samples = args
    return partial_test(agent, target_sector, samples)

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# --- The Sector Class (Unchanged) ---
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
class Sector:
    def __init__(self, target_sector: np.ndarray, palette: list, sector_offset: Tuple[int, int], config: dict):
        self.target_sector = target_sector
        self.palette = palette
        self.sector_offset = sector_offset
        self.config = config
        
        self.img_size = (target_sector.shape[1], target_sector.shape[0])
        self.population = [
            Agent(
                img_size=self.img_size,
                palette=self.palette,
                shapes_per_agent=self.config['shapes_per_agent'],
                sector_offset=self.sector_offset
            ) for _ in range(self.config['n_agents_per_sector'])
        ]

    def step(self, executor: ProcessPoolExecutor, annealing_threshold: float):
        samples = sample_pixels(self.img_size[0], self.img_size[1], self.config['n_samples'])
        
        tasks_args = [(agent, self.target_sector, samples) for agent in self.population]
        future_to_agent = {executor.submit(_parallel_test_wrapper, args): agent for args, agent in zip(tasks_args, self.population)}
        
        agent_scores = {}
        for future in as_completed(future_to_agent):
            agent = future_to_agent[future]
            try:
                agent_scores[agent.id] = future.result()
            except Exception:
                agent_scores[agent.id] = float('inf')

        # --- ELITISM: Find the best agent and protect it ---
        best_agent_id = min(agent_scores, key=agent_scores.get)
        
        # Activate/deactivate agents
        for agent in self.population:
            if agent.id == best_agent_id:
                agent.is_active = True # Protect the elite agent
                continue
            
            score = agent_scores.get(agent.id, float('inf'))
            agent.is_active = score < annealing_threshold

        inactive_agents = [agent for agent in self.population if not agent.is_active]
        random.shuffle(inactive_agents)

        for agent_to_update in inactive_agents:
            communicating_agent = random.choice(self.population)
            if communicating_agent.is_active:
                agent_to_update.shapes = list(communicating_agent.shapes)
                agent_to_update.mutate()
            else:
                agent_to_update._create_random_shapes()
    
    def get_best_agent(self) -> Agent:
        if not self.population:
            return None
            
        best_agent = min(
            self.population,
            key=lambda agent: full_fitness(agent.shapes, self.target_sector, self.config['background_color'])
        )
        return best_agent

# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
# --- The Main DiffusionSearch Class (Corrected Logic) ---
# ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
class DiffusionSearch:
    def __init__(self, target, palette, n_sectors_x=5, n_sectors_y=5, sector_overlap=0.25, **config):
        self.target = target
        self.palette = palette
        self.img_size = (target.shape[1], target.shape[0])
        self.config = config
        self.background_color = self.config.get('background_color', (255, 255, 255, 255))

        self.sectors: List[Sector] = []
        self.executor = ProcessPoolExecutor()

        # ▼▼▼ NEW AND CORRECTED SECTOR CREATION LOGIC ▼▼▼
        
        # 1. Calculate the base size of a "cell" if there were no overlap
        cell_w = self.img_size[0] / n_sectors_x
        cell_h = self.img_size[1] / n_sectors_y

        # 2. Calculate the actual, larger size of our sectors including overlap
        sector_w = int(cell_w * (1 + sector_overlap))
        sector_h = int(cell_h * (1 + sector_overlap))
        
        print(f"Creating {n_sectors_x}x{n_sectors_y} overlapping sectors of size {sector_w}x{sector_h}...")

        # 3. Find the center points for each cell
        center_points_x = np.linspace(cell_w / 2, self.img_size[0] - cell_w / 2, n_sectors_x)
        center_points_y = np.linspace(cell_h / 2, self.img_size[1] - cell_h / 2, n_sectors_y)
        
        for cy in center_points_y:
            for cx in center_points_x:
                # 4. Calculate the top-left corner (offset) for this oversized sector
                x1 = int(cx - sector_w / 2)
                y1 = int(cy - sector_h / 2)

                # 5. Clamp coordinates to stay within the image boundaries
                x1_clamped = max(0, x1)
                y1_clamped = max(0, y1)
                x2_clamped = min(self.img_size[0], x1 + sector_w)
                y2_clamped = min(self.img_size[1], y1 + sector_h)
                
                offset = (x1_clamped, y1_clamped)
                target_crop = target[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

                if target_crop.shape[0] == 0 or target_crop.shape[1] == 0:
                    continue
                
                sector = Sector(
                    target_sector=target_crop,
                    palette=self.palette,
                    sector_offset=offset,
                    config=self.config
                )
                self.sectors.append(sector)

    def __del__(self):
        if self.executor:
            self.executor.shutdown()

    def _calculate_annealing_threshold(self, current_iteration, total_iterations):
        start_threshold = self.config.get('start_threshold', 255.0)
        end_threshold = self.config.get('end_threshold', 0.1)
        progress = current_iteration / total_iterations
        decay_factor = np.exp(-2 * progress)
        return end_threshold + (start_threshold - end_threshold) * decay_factor

    def run(self, iterations):
        total_sectors = len(self.sectors)
        print(f"Starting SDS with {total_sectors} sectors for {iterations} iterations...")
        
        for i in range(iterations):
            threshold_used = self._calculate_annealing_threshold(i, iterations)
            
            for sector in self.sectors:
                sector.step(self.executor, threshold_used)
            
            print(f"Completed iteration {i + 1}/{iterations} (Threshold: {threshold_used:.2f})")
            
        print("SDS run complete.")

    def get_final_shapes(self) -> List:
        """Collects the shapes from the best agent of each sector."""
        print("Finding best agent in each sector...")
        final_shapes = []
        for i, sector in enumerate(self.sectors):
            print(f"  - Evaluating Sector {i+1}/{len(self.sectors)}...")
            best_agent = sector.get_best_agent()
            if best_agent:
                final_shapes.extend(best_agent.shapes)
        return final_shapes