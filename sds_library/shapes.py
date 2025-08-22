# sds_library/shapes.py
import abc
import numpy as np
import random
from typing import Tuple, List

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
    def random_init(self, sector_width: int, sector_height: int, sector_offset: Tuple[int, int] = (0,0)):
        pass

    @abc.abstractmethod
    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        pass

    # ▼▼▼ ADD THIS NEW ABSTRACT METHOD ▼▼▼
    @abc.abstractmethod
    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.1):
        """Applies a small, random change to the shape's properties."""
        pass


class Circle(Shape):
    def random_init(self, sector_width: int, sector_height: int, sector_offset: Tuple[int, int] = (0,0)):
        max_radius = min(sector_width, sector_height) // 4
        self.radius = float(np.random.randint(5, max(6, max_radius)))
        local_cx = float(np.random.randint(self.radius, sector_width - self.radius))
        local_cy = float(np.random.randint(self.radius, sector_height - self.radius))
        self.center = (local_cx + sector_offset[0], local_cy + sector_offset[1])

    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 0, np.array([self.center[0], self.center[1], self.radius**2], dtype=np.float32)

    # ▼▼▼ IMPLEMENT MUTATE FOR CIRCLE ▼▼▼
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


class Rectangle(Shape):
    def random_init(self, sector_width: int, sector_height: int, sector_offset: Tuple[int, int] = (0,0)):
        max_w, max_h = sector_width // 2, sector_height // 2
        width = float(np.random.randint(10, max(11, max_w)))
        height = float(np.random.randint(10, max(11, max_h)))
        local_x1 = float(np.random.randint(0, sector_width - width))
        local_y1 = float(np.random.randint(0, sector_height - height))
        self.top_left = (local_x1 + sector_offset[0], local_y1 + sector_offset[1])
        self.bottom_right = (self.top_left[0] + width, self.top_left[1] + height)

    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 1, np.array([self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1]], dtype=np.float32)

    # ▼▼▼ IMPLEMENT MUTATE FOR RECTANGLE ▼▼▼
    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.1):
        width = self.bottom_right[0] - self.top_left[0]
        height = self.bottom_right[1] - self.top_left[1]
        
        # Adjust size and position
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        
        x1 += (np.random.rand() - 0.5) * width * mutation_strength
        y1 += (np.random.rand() - 0.5) * height * mutation_strength
        x2 += (np.random.rand() - 0.5) * width * mutation_strength
        y2 += (np.random.rand() - 0.5) * height * mutation_strength

        self.top_left = (np.clip(x1, 0, img_width), np.clip(y1, 0, img_height))
        self.bottom_right = (np.clip(x2, 0, img_width), np.clip(y2, 0, img_height))


class Triangle(Shape):
    def random_init(self, sector_width: int, sector_height: int, sector_offset: Tuple[int, int] = (0,0)):
        p1_local = (float(np.random.randint(0, sector_width)), float(np.random.randint(0, sector_height)))
        p2_local = (float(np.random.randint(0, sector_width)), float(np.random.randint(0, sector_height)))
        p3_local = (float(np.random.randint(0, sector_width)), float(np.random.randint(0, sector_height)))
        self.p1 = (p1_local[0] + sector_offset[0], p1_local[1] + sector_offset[1])
        self.p2 = (p2_local[0] + sector_offset[0], p2_local[1] + sector_offset[1])
        self.p3 = (p3_local[0] + sector_offset[0], p3_local[1] + sector_offset[1])
        
    def get_numba_data(self) -> Tuple[int, np.ndarray]:
        return 2, np.array([self.p1[0], self.p1[1], self.p2[0], self.p2[1], self.p3[0], self.p3[1]], dtype=np.float32)

    # ▼▼▼ IMPLEMENT MUTATE FOR TRIANGLE ▼▼▼
    def mutate(self, img_width: int, img_height: int, mutation_strength: float = 0.1):
        # Pick one vertex and move it slightly
        points = [list(self.p1), list(self.p2), list(self.p3)]
        point_to_mutate = random.choice(points)

        move_x = (np.random.rand() - 0.5) * img_width * mutation_strength
        move_y = (np.random.rand() - 0.5) * img_height * mutation_strength
        
        point_to_mutate[0] = np.clip(point_to_mutate[0] + move_x, 0, img_width)
        point_to_mutate[1] = np.clip(point_to_mutate[1] + move_y, 0, img_height)
        
        self.p1, self.p2, self.p3 = tuple(points[0]), tuple(points[1]), tuple(points[2])