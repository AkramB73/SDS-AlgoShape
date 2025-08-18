import abc
import numpy as np
import random
from typing import Tuple,List

RGBA = Tuple[int, int, int, int]
palette_form =  List[Tuple[int, int, int]]

class Shape(abc.ABC):
    def __init__(self, palette):
        rgb_color = random.choice(palette)
        
        # Random alpha value allows for transparency and blending
        alpha = np.random.randint(50, 150)
        self._color: RGBA = rgb_color + (alpha,)    

    @property
    def color(self) -> RGBA:
        return self._color

    @abc.abstractmethod
    def random_init(self, img_width: int, img_height: int):
        pass

    @abc.abstractmethod
    def contains(self, x: int, y: int) -> bool:
        pass

# Shape Implementations 

class Circle(Shape):
    def __init__(self,palette):
        super().__init__(palette)
        self.center = (0, 0)
        self.radius = 0

    def random_init(self, img_width: int, img_height: int):
        # Kept it at half the max radius possible to stop a shape from dominating. 
        max_radius = min(img_width, img_height) // 4
        self.radius = np.random.randint(5, max_radius)
        # Sets the smallest size of radius being 5 pixels.
        self.center = (
            np.random.randint(0, img_width),
            np.random.randint(0, img_height)
        )

    def contains(self, x: int, y: int) -> bool:
        cx, cy = self.center
        return (x - cx)**2 + (y - cy)**2 < self.radius**2

class Rectangle(Shape):
    # An axis-aligned rectangle defined by its top-left corner and size.
    def __init__(self,palette):
        super().__init__(palette)
        self.top_left = (0, 0)
        self.size = (0, 0) # (width, height)

    def random_init(self, img_width: int, img_height: int):
        max_w = img_width // 2
        max_h = img_height // 2
        
        width = np.random.randint(10, max_w)
        height = np.random.randint(10, max_h)
        self.size = (width, height)
        
        x = np.random.randint(0, img_width - width)
        y = np.random.randint(0, img_height - height)
        self.top_left = (x, y)

    def contains(self, x: int, y: int) -> bool:
        x1, y1 = self.top_left
        w, h = self.size
        x2, y2 = x1 + w, y1 + h
        return (x1 <= x < x2) and (y1 <= y < y2)

class Triangle(Shape):
    def __init__(self,palette):
        super().__init__(palette)
        self.p1 = (0, 0)
        self.p2 = (0, 0)
        self.p3 = (0, 0)

    def random_init(self, img_width: int, img_height: int):
        self.p1 = (np.random.randint(0, img_width), np.random.randint(0, img_height))
        self.p2 = (np.random.randint(0, img_width), np.random.randint(0, img_height))
        self.p3 = (np.random.randint(0, img_width), np.random.randint(0, img_height))

    def contains(self, x: int, y: int) -> bool:
        # Barycentric technique to check if point is inside triangle
        p = (x, y)

        # A point is inside a triangle if it's on the same side of all three edges.
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, self.p1, self.p2)
        d2 = sign(p, self.p2, self.p3)
        d3 = sign(p, self.p3, self.p1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)