# sds_library/agent.py

import numpy as np
import random
from typing import List, Tuple
import uuid

# (Imports remain the same)
from .shapes import Shape, Circle, Rectangle, Triangle
from .evaluator import render_pixel_jit

class Agent:
    # (__init__ method remains the same)
    def __init__(self, img_size: Tuple[int, int], palette: list, shapes_per_agent: int):
        self.img_size = img_size
        self.palette = palette
        self.shapes_per_agent = shapes_per_agent
        self.shapes: List[Shape] = []
        self.is_active: bool = False
        self.id = uuid.uuid4()
        
        self.shape_params = np.empty((0, 7), dtype=np.float32)
        self.shape_colors = np.empty((0, 4), dtype=np.float32)

        self._create_random_shapes()

    # (_create_random_shapes and _prepare_data_for_numba remain the same)
    def _create_random_shapes(self):
        self.shapes = []
        img_width, img_height = self.img_size
        for _ in range(self.shapes_per_agent):
            shape_class = random.choice([Triangle, Circle, Rectangle])
            shape = shape_class(palette=self.palette)
            shape.random_init(img_width, img_height)
            self.shapes.append(shape)
        self._prepare_data_for_numba()

    def _prepare_data_for_numba(self):
        self.shape_params = np.zeros((self.shapes_per_agent, 7), dtype=np.float32)
        self.shape_colors = np.zeros((self.shapes_per_agent, 4), dtype=np.float32)
        for i, shape in enumerate(self.shapes):
            shape_type, params = shape.get_numba_data()
            self.shape_params[i, 0] = shape_type
            self.shape_params[i, 1:1+len(params)] = params
            self.shape_colors[i] = shape.color

    # ▼▼▼ ADD THE NEW MUTATE METHOD HERE ▼▼▼
    def mutate(self):
        """
        Applies small random changes to a few shapes. This is the key to refining solutions.
        """
        # Mutate 10% of the shapes, but at least one
        num_to_mutate = max(1, int(self.shapes_per_agent * 0.1))
        
        indices_to_mutate = random.sample(range(self.shapes_per_agent), num_to_mutate)
        
        for i in indices_to_mutate:
            # Create a new random shape to replace the old one
            img_width, img_height = self.img_size
            shape_class = random.choice([Triangle]) # Only Triangle for now, can be extended
            new_shape = shape_class(palette=self.palette)
            new_shape.random_init(img_width, img_height)
            self.shapes[i] = new_shape
            
        # IMPORTANT: After mutating, you must rebuild the Numba data arrays
        self._prepare_data_for_numba()

    # (render_pixel method remains the same)
    def render_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        background_color = np.array((255, 255, 255, 255), dtype=np.float32)
        final_color_arr = render_pixel_jit(x, y, self.shape_params, self.shape_colors, background_color)
        return tuple(map(int, final_color_arr))