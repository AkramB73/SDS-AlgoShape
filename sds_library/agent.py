
import numpy as np
import random
from typing import List, Tuple, Type
import uuid

from .shapes import Shape, Triangle

class Agent:
    def __init__(self, img_size: Tuple[int, int], palette: list, shapes_per_agent: int, shape_types: List[Type[Shape]]):
        self.img_size = img_size
        self.palette = palette
        self.shapes_per_agent = shapes_per_agent
        self.shapes: List[Shape] = []
        self.id = uuid.uuid4()
        
        self.shape_types = shape_types
        
        self.shape_params = np.empty((0, 7), dtype=np.float32)
        self.shape_colors = np.empty((0, 4), dtype=np.uint8)

        self._create_random_shapes()

    def _create_random_shapes(self):
        self.shapes = []
        img_width, img_height = self.img_size
        for _ in range(self.shapes_per_agent):
            shape_class = random.choice(self.shape_types) 
            shape = shape_class(palette=self.palette)
            shape.random_init(img_width, img_height)
            self.shapes.append(shape)
        self._prepare_data()

    def _prepare_data(self):
        self.shape_params = np.zeros((self.shapes_per_agent, 7), dtype=np.float32)
        self.shape_colors = np.zeros((self.shapes_per_agent, 4), dtype=np.uint8)
        for i, shape in enumerate(self.shapes):
            shape_type, params = shape.get_numba_data()
            padded_params = np.zeros(6)
            padded_params[:len(params)] = params
            
            self.shape_params[i, 0] = shape_type
            self.shape_params[i, 1:] = padded_params
            self.shape_colors[i] = shape.color

    def mutate(self):
        num_to_mutate = max(1, int(self.shapes_per_agent * 0.3))
        indices_to_mutate = random.sample(range(self.shapes_per_agent), num_to_mutate)
        
        img_width, img_height = self.img_size
        for i in indices_to_mutate:
            self.shapes[i].mutate(img_width, img_height)

        self._prepare_data()