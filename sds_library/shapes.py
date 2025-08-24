# sds_library/shapes.py

import abc
import numpy as np
import random
from typing import Tuple, List

# Type aliases remain the same
RGBA = Tuple[int, int, int, int]
Palette = List[Tuple[int, int, int]]

class Shape(abc.ABC):
    def __init__(self, palette: Palette):
        rgb_color = random.choice(palette)
        # Use a semi-transparent alpha for a nice layered look
        alpha = int(np.random.randint(50, 150))
        self._color: RGBA = rgb_color + (alpha,)

    @property
    def color(self) -> RGBA:
        return self._color

    def mutate_color(self, mutation_strength: int = 15):
        """Slightly alters the R, G, B, and Alpha channels of the shape."""
        r, g, b, a = self._color
        
        r += np.random.randint(-mutation_strength, mutation_strength + 1)
        g += np.random.randint(-mutation_strength, mutation_strength + 1)
        b += np.random.randint(-mutation_strength, mutation_strength + 1)
        a += np.random.randint(-mutation_strength, mutation_strength + 1)
        
        # Clamp the values to the valid 0-255 range
        self._color = (
            np.clip(r, 0, 255),
            np.clip(g, 0, 255),
            np.clip(b, 0, 255),
            np.clip(a, 0, 255)
        )

    @abc.abstractmethod
    def random_init(self, img_width: int, img_height: int):
        pass

    @abc.abstractmethod
    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        pass
        
    @abc.abstractmethod
    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.1):
        """Applies a small, random change to the shape's properties."""
        pass


class Triangle(Shape):
    def random_init(self, img_width: int, img_height: int):
        img_width *= 1.02
        img_height *= 1.02
        self.p1 = (float(np.random.randint(0, img_width)), float(np.random.randint(0, img_height)))
        self.p2 = (float(np.random.randint(0, img_width)), float(np.random.randint(0, img_height)))
        self.p3 = (float(np.random.randint(0, img_width)), float(np.random.randint(0, img_height)))
        
    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 2, np.array([self.p1[0], self.p1[1], self.p2[0], self.p2[1], self.p3[0], self.p3[1]], dtype=np.float32)

    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.1):
        # Pick one vertex and move it slightly
        points = [list(self.p1), list(self.p2), list(self.p3)]
        point_to_mutate = random.choice(points)

        move_x = (np.random.rand() - 0.5) * img_width * mutation_strength
        move_y = (np.random.rand() - 0.5) * img_height * mutation_strength
        
        point_to_mutate[0] = np.clip(point_to_mutate[0] + move_x, 0, img_width)
        point_to_mutate[1] = np.clip(point_to_mutate[1] + move_y, 0, img_height)
        
        self.p1, self.p2, self.p3 = tuple(points[0]), tuple(points[1]), tuple(points[2])
        self.mutate_color()



class Circle(Shape):
    def random_init(self, img_width: int, img_height: int):
        max_radius = min(img_width, img_height) // 4
        self.radius = float(np.random.randint(5, max(6, max_radius)))
        self.center = (
            float(np.random.randint(self.radius, img_width - self.radius)),
            float(np.random.randint(self.radius, img_height - self.radius))
        )

    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 0, np.array([self.center[0], self.center[1], self.radius**2], dtype=np.float32)

    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.1):
        # Adjust radius by a small amount
        radius_change = (np.random.rand() - 0.5) * self.radius * mutation_strength
        self.radius = np.clip(self.radius + radius_change, 2, min(img_width, img_height) / 2)

        # Move center by a small amount
        move_x = (np.random.rand() - 0.5) * img_width * mutation_strength
        move_y = (np.random.rand() - 0.5) * img_height * mutation_strength
        self.center = (
            np.clip(self.center[0] + move_x, 0, img_width),
            np.clip(self.center[1] + move_y, 0, img_height)
        )
        self.mutate_color()


class Rectangle(Shape):
    def random_init(self, img_width: int, img_height: int):
        max_w, max_h = img_width // 2, img_height // 2
        width = float(np.random.randint(10, max(11, max_w)))
        height = float(np.random.randint(10, max(11, max_h)))
        x1 = float(np.random.randint(0, img_width - width))
        y1 = float(np.random.randint(0, img_height - height))
        self.top_left = (x1, y1)
        self.bottom_right = (x1 + width, y1 + height)

    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 1, np.array([self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1]], dtype=np.float32)

    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.2):
        width = self.bottom_right[0] - self.top_left[0]
        height = self.bottom_right[1] - self.top_left[1]
        
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        
        x1 += (np.random.rand() - 0.5) * width * mutation_strength
        y1 += (np.random.rand() - 0.5) * height * mutation_strength
        x2 += (np.random.rand() - 0.5) * width * mutation_strength
        y2 += (np.random.rand() - 0.5) * height * mutation_strength

        self.top_left = (np.clip(x1, 0, img_width), np.clip(y1, 0, img_height))
        self.bottom_right = (np.clip(x2, 0, img_width), np.clip(y2, 0, img_height))
        self.mutate_color()
