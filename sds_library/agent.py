# sds_library/agent.py

import numpy as np
import random
from typing import List, Tuple
import uuid

from .shapes import Shape, Circle, Rectangle, Triangle
from .evaluator import render_pixel_jit

class Agent:
    def __init__(self, img_size: Tuple[int, int], palette: list, shapes_per_agent: int, sector_offset: Tuple[int, int] = (0, 0), config: dict = {}):
        self.img_size = img_size # This is the size of the SECTOR
        self.palette = palette
        self.shapes_per_agent = shapes_per_agent
        self.sector_offset = sector_offset
        self.config = config # Store the main configuration
        self.shapes: List[Shape] = []
        self.is_active: bool = False
        self.id = uuid.uuid4()
        
        self.shape_params = np.empty((0, 7), dtype=np.float32)
        self.shape_colors = np.empty((0, 4), dtype=np.float32)

        self._create_random_shapes()

    def _create_random_shapes(self):
        """Creates a new set of random shapes and prepares the data for Numba."""
        self.shapes = []
        sector_width, sector_height = self.img_size
        for _ in range(self.shapes_per_agent):
            # You can change this list to control which shapes are used
            shape_class = random.choice([Triangle]) # Circle, Rectangle, Triangle
            shape = shape_class(palette=self.palette)
            shape.random_init(sector_width, sector_height, self.sector_offset)
            self.shapes.append(shape)
        
        self._prepare_data_for_numba()

    def _prepare_data_for_numba(self):
        """
        Converts the list of shape objects into simple NumPy arrays
        that can be passed to Numba with zero overhead.
        """
        self.shape_params = np.zeros((self.shapes_per_agent, 7), dtype=np.float32)
        self.shape_colors = np.zeros((self.shapes_per_agent, 4), dtype=np.float32)

        for i, shape in enumerate(self.shapes):
            shape_type, params = shape.get_numba_data()
            self.shape_params[i, 0] = shape_type
            self.shape_params[i, 1:1+len(params)] = params
            self.shape_colors[i] = shape.color

    def mutate(self):
        """
        Applies small, refining changes to a few shapes instead of replacing them.
        This is the key to finding structure in the image.
        """
        # Mutate 20% of the shapes, but at least one
        num_to_mutate = max(1, int(self.shapes_per_agent * 0.3))
        
        indices_to_mutate = random.sample(range(self.shapes_per_agent), num_to_mutate)
        
        # Get the overall image dimensions from the config for mutation boundaries
        # This is a fallback in case the config isn't passed, but it should be.
        full_width = self.config.get('processing_width', self.img_size[0])
        full_height = self.config.get('processing_height', self.img_size[1])

        for i in indices_to_mutate:
            shape_to_mutate = self.shapes[i]
            # Call the shape's own internal mutate method
            shape_to_mutate.mutate(full_width, full_height)

        # IMPORTANT: After mutating, you must rebuild the Numba data arrays
        self._prepare_data_for_numba()

    def render_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        Calculates the color of a single pixel by calling the fast JIT function.
        """
        background_color = np.array((255, 255, 255, 255), dtype=np.float32)
        
        # We need to adjust the pixel coordinates to be relative to the sector for evaluation
        local_x = x - self.sector_offset[0]
        local_y = y - self.sector_offset[1]
        
        final_color_arr = render_pixel_jit(local_x, local_y, self.shape_params, self.shape_colors, background_color)
        
        return tuple(map(int, final_color_arr))