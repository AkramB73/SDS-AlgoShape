# sds_library/evaluator.py

import numpy as np
import metalcompute as mc
from typing import List, Tuple

from .metal_kernels import EVALUATOR_KERNEL_CODE
from .agent import Agent
from .shapes import Shape
from .spatial_grid import SpatialGrid

class MetalEvaluator:
    def __init__(self, target_image: np.ndarray, shapes_per_agent: int, n_samples: int, n_agents: int):
        print("Initializing Metal evaluator (Final Architecture)...")
        
        self.device = mc.Device()
        self.kernel = self.device.kernel(EVALUATOR_KERNEL_CODE).function("evaluate_all_agents_with_grids")
        
        self.img_height, self.img_width, _ = target_image.shape
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples
        self.n_agents = n_agents

        self.grid = SpatialGrid(self.img_width, self.img_height)

        # Constant data buffers
        self.target_image_buffer = self.device.buffer(target_image.tobytes())
        self.shapes_per_agent_buffer = self.device.buffer(np.uint32(self.shapes_per_agent).tobytes())
        self.n_samples_buffer = self.device.buffer(np.uint32(self.n_samples).tobytes())
        self.image_width_buffer = self.device.buffer(np.uint32(self.img_width).tobytes())
        self.grid_size_buffer = self.device.buffer(np.uint32(self.grid.grid_size).tobytes())
        self.cell_width_buffer = self.device.buffer(np.float32(self.grid.cell_width).tobytes())
        self.cell_height_buffer = self.device.buffer(np.float32(self.grid.cell_height).tobytes())

        # Pre-allocate buffers for data with predictable sizes
        pop_params_size = n_agents * shapes_per_agent * 7 * 4
        pop_colors_size = n_agents * shapes_per_agent * 4 * 4
        self.pop_params_buffer = self.device.buffer(pop_params_size)
        self.pop_colors_buffer = self.device.buffer(pop_colors_size)

        blocks_size = n_samples * 2 * 4
        self.blocks_buffer = self.device.buffer(blocks_size)

        # Pre-allocate the output buffer, assuming a max block size to be safe
        max_block_size = 10 
        pixels_per_agent = n_samples * (max_block_size * max_block_size)
        total_pixels = n_agents * pixels_per_agent
        self.pixel_errors_buffer = self.device.buffer(total_pixels * 4)

        print(f"Metal device initialized for final architecture.")

    def evaluate(self, population: List[Agent], blocks: np.ndarray) -> np.ndarray:
        n_agents = len(population)
        if n_agents == 0:
            return np.array([])

        pop_params = np.array([agent.shape_params for agent in population], dtype=np.float32)
        pop_colors_int = np.array([agent.shape_colors for agent in population], dtype=np.uint8)
        pop_colors_float = pop_colors_int.astype(np.float32) / 255.0

        all_grid_indices = []
        all_cell_offsets = []
        agent_grid_offsets = np.zeros((n_agents, 2), dtype=np.int32)
        
        current_indices_offset = 0
        current_cells_offset = 0

        for i, agent in enumerate(population):
            self.grid.add_shapes(agent.shapes)
            grid_indices, cell_offsets = self.grid.get_flattened_data()
            
            all_grid_indices.append(grid_indices)
            all_cell_offsets.append(cell_offsets)
            agent_grid_offsets[i] = [current_indices_offset, current_cells_offset]
            
            current_indices_offset += len(grid_indices)
            current_cells_offset += len(cell_offsets)
        
        mega_grid_indices = np.concatenate(all_grid_indices) if all_grid_indices else np.array([], dtype=np.int32)
        mega_cell_offsets = np.concatenate(all_cell_offsets) if all_cell_offsets else np.array([], dtype=np.int32)

        # --- MODIFIED: Create grid buffers dynamically with the exact size ---
        mega_grid_indices_buffer = self.device.buffer(mega_grid_indices.tobytes())
        mega_cell_offsets_buffer = self.device.buffer(mega_cell_offsets.tobytes())
        agent_grid_offsets_buffer = self.device.buffer(agent_grid_offsets.tobytes())

        # Write data to pre-allocated buffers
        memoryview(self.pop_params_buffer).cast('f')[:pop_params.size] = pop_params.flatten()
        memoryview(self.pop_colors_buffer).cast('f')[:pop_colors_float.size] = pop_colors_float.flatten()
        memoryview(self.blocks_buffer).cast('i')[:blocks.size] = blocks.flatten()
        
        block_size_val = blocks.shape[1]
        block_size_arr = np.uint32(block_size_val)

        pixels_per_block = block_size_val * block_size_val
        pixels_per_agent = self.n_samples * pixels_per_block
        total_threads = n_agents * pixels_per_agent

        handle = self.kernel(
            total_threads,
            self.pop_params_buffer,
            self.pop_colors_buffer,
            mega_grid_indices_buffer, # Use dynamic buffer
            mega_cell_offsets_buffer, # Use dynamic buffer
            agent_grid_offsets_buffer,  # Use dynamic buffer
            self.target_image_buffer,
            self.blocks_buffer,
            self.shapes_per_agent_buffer,
            self.n_samples_buffer,
            block_size_arr,
            self.image_width_buffer,
            self.grid_size_buffer,
            self.cell_width_buffer,
            self.cell_height_buffer,
            self.pixel_errors_buffer
        )
        del handle
        
        errors_view = memoryview(self.pixel_errors_buffer).cast('f')
        all_pixel_errors = np.array(errors_view[:total_threads])
        
        errors_by_agent = all_pixel_errors.reshape((n_agents, pixels_per_agent))
        total_error_per_agent = np.sum(errors_by_agent, axis=1)
        final_scores = total_error_per_agent / pixels_per_agent

        return final_scores


def full_fitness(agent_shapes: List['Shape'], target_image: np.ndarray, background_color: tuple) -> float:
    # This function remains unchanged.
    import cv2
    from .shapes import Circle, Rectangle, Triangle
    h, w, _ = target_image.shape
    canvas = np.full((h, w, 4), background_color, dtype=np.float32)
    for shape in agent_shapes:
        if isinstance(shape, Circle):
            fx1, fy1 = shape.center[0] - shape.radius, shape.center[1] - shape.radius
            fx2, fy2 = shape.center[0] + shape.radius, shape.center[1] + shape.radius
        elif isinstance(shape, Rectangle):
            fx1, fy1 = shape.top_left[0], shape.top_left[1]
            fx2, fy2 = shape.bottom_right[0], shape.bottom_right[1]
        elif isinstance(shape, Triangle):
            points = np.array([shape.p1, shape.p2, shape.p3])
            fx1, fy1 = points.min(axis=0)
            fx2, fy2 = points.max(axis=0)
        else:
            continue
        x1, y1 = max(0, int(fx1)), max(0, int(fy1))
        x2, y2 = min(w, int(np.ceil(fx2))), min(h, int(np.ceil(fy2)))
        if x1 >= x2 or y1 >= y2:
            continue
        box_w, box_h = x2 - x1, y2 - y1
        shape_layer = np.zeros((box_h, box_w, 4), dtype=np.float32)
        shape_color_bgra = tuple(map(int, (shape.color[2], shape.color[1], shape.color[0], shape.color[3])))
        if isinstance(shape, Circle):
            center_local = (int(shape.center[0] - x1), int(shape.center[1] - y1))
            cv2.circle(shape_layer, center_local, int(shape.radius), shape_color_bgra, -1)
        elif isinstance(shape, Rectangle):
            cv2.rectangle(shape_layer, (0, 0), (box_w, box_h), shape_color_bgra, -1)
        elif isinstance(shape, Triangle):
            points_local = np.array([(int(p[0] - x1), int(p[1] - y1)) for p in [shape.p1, shape.p2, shape.p3]], dtype=np.int32)
            cv2.fillPoly(shape_layer, [points_local], shape_color_bgra)
        canvas_region = canvas[y1:y2, x1:x2]
        alpha = shape_layer[:, :, 3:4] / 255.0
        canvas[y1:y2, x1:x2] = shape_layer * alpha + canvas_region * (1.0 - alpha)
    rendered_image_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)
    diff = np.subtract(target_image.astype(np.float32), rendered_image_uint8.astype(np.float32))
    rmse = np.sqrt(np.mean(np.square(diff)))
    return float(rmse)