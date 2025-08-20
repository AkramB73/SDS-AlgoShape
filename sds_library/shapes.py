# sds_library/shapes.py

import abc
import numpy as np
import random
from typing import Tuple, List

# (Type aliases remain the same)
RGBA = Tuple[int, int, int, int]
Palette = List[Tuple[int, int, int]]

class Shape(abc.ABC):
    def __init__(self, palette: Palette):
        rgb_color = random.choice(palette)
        alpha = int(np.random.randint(50, 150))
        self._color: RGBA = rgb_color + (alpha,)

    @property
    def color(self) -> RGBA:
        return self._color

    @abc.abstractmethod
    def random_init(self, img_width: int, img_height: int):
        pass

    @abc.abstractmethod
    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        pass

class Circle(Shape):
    def random_init(self, img_width: int, img_height: int):
        max_radius = min(img_width, img_height) // 4
        self.radius = float(np.random.randint(5, max(6, max_radius)))
        # Ensure center is placed such that the circle is fully in-bounds
        self.center = (
            float(np.random.randint(self.radius, img_width - self.radius)),
            float(np.random.randint(self.radius, img_height - self.radius))
        )

    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 0, np.array([self.center[0], self.center[1], self.radius**2], dtype=np.float32)

class Rectangle(Shape):
    def random_init(self, img_width: int, img_height: int):
        max_w, max_h = img_width // 2, img_height // 2
        width = float(np.random.randint(10, max(11, max_w)))
        height = float(np.random.randint(10, max(11, max_h)))
        # Ensure top-left is placed such that the rect is fully in-bounds
        x1 = float(np.random.randint(0, img_width - width))
        y1 = float(np.random.randint(0, img_height - height))
        self.top_left = (x1, y1)
        self.bottom_right = (x1 + width, y1 + height)

    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 1, np.array([self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1]], dtype=np.float32)

class Triangle(Shape):
    def random_init(self, img_width: int, img_height: int):
        img_height *= 1.1
        img_width *= 1.1
        self.p1 = (float(np.random.randint(0, img_width)), float(np.random.randint(0, img_height)))
        self.p2 = (float(np.random.randint(0, img_width)), float(np.random.randint(0, img_height)))
        self.p3 = (float(np.random.randint(0, img_width)), float(np.random.randint(0, img_height)))
        
    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 2, np.array([self.p1[0], self.p1[1], self.p2[0], self.p2[1], self.p3[0], self.p3[1]], dtype=np.float32)