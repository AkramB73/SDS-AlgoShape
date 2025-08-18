import numpy as np
import random
from typing import List, Tuple
import uuid 

from .shapes import Shape, Circle, Rectangle, Triangle

RGBA = Tuple[int, int, int, int]
ImgSize = Tuple[int, int]

AVAILABLE_SHAPES = [Circle, Rectangle, Triangle]

class Agent:
    def __init__(self, img_size: ImgSize, palette: list, shapes_per_agent: int):
        self.img_size = img_size
        self.palette = palette
        self.shapes_per_agent = shapes_per_agent
        self.shapes: List[Shape] = []
        self.is_active: bool = False
        # Assign a stable, unique ID for multiprocessing 
        self.id = uuid.uuid4()        
        
        self._create_random_shapes()

    def _create_random_shapes(self):
        self.shapes = []
        img_width, img_height = self.img_size
        for _ in range(self.shapes_per_agent):

            shape_class = random.choice(AVAILABLE_SHAPES)
            shape = shape_class(palette=self.palette)
            shape.random_init(img_width, img_height)
            self.shapes.append(shape)

    def render_pixel(self, x: int, y: int) -> RGBA:

        final_color = np.array((255, 255, 255, 255), dtype=np.float32) # Start with white background

        for shape in self.shapes:
            if shape.contains(x, y):
                fg_color = np.array(shape.color, dtype=np.float32)
                bg_color = final_color

                fg_alpha = fg_color[3] / 255.0
                bg_alpha = bg_color[3] / 255.0

                # Blend the RGB channels
                final_color[:3] = (fg_color[:3] * fg_alpha) + (bg_color[:3] * (1.0 - fg_alpha))

                # For alpha channel
                final_color[3] = (fg_alpha + bg_alpha * (1.0 - fg_alpha)) * 255.0

        return tuple(final_color.astype(np.uint8))