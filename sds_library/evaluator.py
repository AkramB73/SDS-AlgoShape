# sds_library/evaluator.py

import numpy as np
import metalcompute as mc
from typing import List, Tuple

from .metal_kernels import EVALUATOR_KERNEL_CODE
from .agent import Agent

class MetalEvaluator:
    def __init__(self, target_image: np.ndarray, shapes_per_agent: int, n_samples: int):
        print("Initializing Metal evaluator...")
        
        self.device = mc.Device()
        self.kernel = self.device.kernel(EVALUATOR_KERNEL_CODE).function("evaluate_population")
        
        self.img_height, self.img_width, _ = target_image.shape
        self.shapes_per_agent = shapes_per_agent
        self.n_samples = n_samples

        self.target_image_buffer = self.device.buffer(target_image.tobytes())
        self.shapes_per_agent_buffer = self.device.buffer(np.uint32(self.shapes_per_agent).tobytes())
        self.n_samples_buffer = self.device.buffer(np.uint32(self.n_samples).tobytes())
        self.image_width_buffer = self.device.buffer(np.uint32(self.img_width).tobytes())

        scores_size = 50 * 4 
        self.scores_buffer = self.device.buffer(scores_size)

        print("Metal device and kernel initialized successfully.")


    def evaluate(self, population: List[Agent], blocks: np.ndarray) -> np.ndarray:
        n_agents = len(population)
        if n_agents == 0:
            return np.array([])

        pop_params = np.array([agent.shape_params for agent in population], dtype=np.float32).flatten()
        pop_colors_int = np.array([agent.shape_colors for agent in population], dtype=np.uint8)
        pop_colors_float = pop_colors_int.astype(np.float32) / 255.0
        pop_colors = pop_colors_float.flatten()
        
        block_size_val = blocks.shape[1] if blocks.ndim > 1 and blocks.shape[1] > 0 else 0
        block_size_arr = np.uint32(block_size_val)

        handle = self.kernel(
            n_agents,
            pop_params,
            pop_colors,
            self.target_image_buffer,
            blocks,
            block_size_arr,
            self.shapes_per_agent_buffer,
            self.n_samples_buffer,
            self.image_width_buffer,
            self.scores_buffer
        )
        
        del handle
        
        # --- MODIFIED: Correct way to read data back from the buffer ---
        # Create a memoryview of the buffer and cast it to the correct type.
        scores_view = memoryview(self.scores_buffer).cast('f')
        
        # Create a NumPy array from the view, slicing to the number of agents.
        scores = np.array(scores_view[:n_agents])

        return scores


def full_fitness(agent_shapes: List['Shape'], target_image: np.ndarray, background_color: tuple) -> float:
    import cv2
    from .shapes import Shape, Circle, Rectangle, Triangle

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