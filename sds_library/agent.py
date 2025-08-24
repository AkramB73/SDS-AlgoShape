# sds_library/agent.py

import numpy as np
import random
from typing import List, Tuple
import uuid

from .shapes import Shape, Circle, Rectangle, Triangle
from .evaluator import render_pixel_jit

class Agent:
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

    def _create_random_shapes(self):
        self.shapes = []
        img_width, img_height = self.img_size
        for _ in range(self.shapes_per_agent):
            # You can add Circle and Rectangle back to this list if you want to use them
            shape_class = random.choice([Triangle]) 
            shape = shape_class(palette=self.palette)
            shape.random_init(img_width, img_height)
            self.shapes.append(shape)
        self._prepare_data_for_numba()

    def _prepare_data_for_numba(self):
        self.shape_params = np.zeros((self.shapes_per_agent, 7), dtype=np.float32)
        self.shape_colors = np.zeros((self.shapes_per_agent, 4), dtype=np.float32)
        for i, shape in enumerate(self.shapes):
            shape_type, params = shape.get_numba_data()
            # Pad the params array if it's smaller than 6 (for circles/rectangles)
            padded_params = np.zeros(6)
            padded_params[:len(params)] = params
            
            self.shape_params[i, 0] = shape_type
            self.shape_params[i, 1:] = padded_params
            self.shape_colors[i] = shape.color

    def mutate(self):
        """
        Applies small, refining changes to a few shapes instead of replacing them.
        """
        # Mutate 20% of the shapes, but at least one
        num_to_mutate = max(1, int(self.shapes_per_agent * 0.3))
        
        indices_to_mutate = random.sample(range(self.shapes_per_agent), num_to_mutate)
        
        img_width, img_height = self.img_size
        for i in indices_to_mutate:
            # Call the shape's own internal mutate method
            self.shapes[i].mutate(img_width, img_height)

        # After mutating, rebuild the Numba data arrays
        self._prepare_data_for_numba()

    def render_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        background_color = np.array((255, 255, 255, 255), dtype=np.float32)
        final_color_arr = render_pixel_jit(x, y, self.shape_params, self.shape_colors, background_color)
        return tuple(map(int, final_color_arr))