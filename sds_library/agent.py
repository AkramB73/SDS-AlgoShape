# sds_library/agent.py

import numpy as np
import random
from typing import List, Tuple
import uuid
from numba import njit

# Import the shape classes
from .shapes import Shape, Circle, Rectangle, Triangle

# --- Numba JIT-Compiled Rendering Function ---
# This function is the new performance hotspot. It takes simple NumPy arrays
# and performs the entire rendering loop for a single pixel in fast, compiled code.
@njit(fastmath=True)
def render_pixel_jit(x: int, y: int, shape_params: np.ndarray, shape_colors: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    """
    Calculates the color of a single pixel by rendering all shapes in a highly
    optimized, JIT-compiled loop.
    """
    final_color = background_color.copy()
    
    for i in range(shape_params.shape[0]):
        params = shape_params[i]
        shape_type = int(params[0])
        
        is_inside = False
        if shape_type == 0: # Circle
            # params[1]=cx, params[2]=cy, params[3]=r_sq
            is_inside = (x - params[1])**2 + (y - params[2])**2 < params[3]
        elif shape_type == 1: # Rectangle
            # params[1]=x1, params[2]=y1, params[3]=x2, params[4]=y2
            is_inside = (params[1] <= x < params[3]) and (params[2] <= y < params[4])
        elif shape_type == 2: # Triangle
            # params[1-6] are the coordinates of the three points
            p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = params[1], params[2], params[3], params[4], params[5], params[6]
            
            # Barycentric coordinate check (sign of cross-products)
            d1 = (x - p3_x) * (p2_y - p3_y) - (p2_x - p3_x) * (y - p3_y)
            d2 = (x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (y - p1_y)
            c = (y - p1_y) * (p2_x - p1_x) - (x - p1_x) * (p2_y - p1_y)
            
            # Check if the point is on the same side of all edges
            if not ((d1 < 0 or d2 < 0 or c < 0) and (d1 > 0 or d2 > 0 or c > 0)):
                 is_inside = True

        if is_inside:
            fg_color = shape_colors[i]
            bg_color = final_color
            
            fg_alpha = fg_color[3] / 255.0
            bg_alpha = bg_color[3] / 255.0
            
            # Alpha blend the RGB channels
            final_color[:3] = fg_color[:3] * fg_alpha + bg_color[:3] * (1.0 - fg_alpha)
            # Combine the alpha channels
            final_color[3] = (fg_alpha + bg_alpha * (1.0 - fg_alpha)) * 255.0

    return final_color


class Agent:
    def __init__(self, img_size: Tuple[int, int], palette: list, shapes_per_agent: int):
        self.img_size = img_size
        self.palette = palette
        self.shapes_per_agent = shapes_per_agent
        self.shapes: List[Shape] = []
        self.is_active: bool = False
        self.id = uuid.uuid4()
        
        # These will hold the simple, Numba-friendly data arrays
        self.shape_params = np.empty((0, 7), dtype=np.float32)
        self.shape_colors = np.empty((0, 4), dtype=np.float32)

        self._create_random_shapes()

    def _create_random_shapes(self):
        """Creates a new set of random shapes and prepares the data for Numba."""
        self.shapes = []
        img_width, img_height = self.img_size
        for _ in range(self.shapes_per_agent):
            shape_class = random.choice([Circle, Rectangle, Triangle])
            shape = shape_class(palette=self.palette)
            shape.random_init(img_width, img_height)
            self.shapes.append(shape)
        
        self._prepare_data_for_numba()

    def _prepare_data_for_numba(self):
        """
        Converts the list of shape objects into two simple NumPy arrays
        that can be passed to Numba with zero overhead.
        This is a crucial optimization.
        """
        # The parameter array needs 7 columns to accommodate the triangle (type + 6 coords)
        self.shape_params = np.zeros((self.shapes_per_agent, 7), dtype=np.float32)
        self.shape_colors = np.zeros((self.shapes_per_agent, 4), dtype=np.float32)

        for i, shape in enumerate(self.shapes):
            shape_type, params = shape.get_numba_data()
            self.shape_params[i, 0] = shape_type
            self.shape_params[i, 1:1+len(params)] = params
            self.shape_colors[i] = shape.color

    def render_pixel(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        Calculates the color of a single pixel by calling the fast JIT function.
        """
        background_color = np.array((255, 255, 255, 255), dtype=np.float32)
        
        # Call the JIT function with the pre-prepared data arrays
        final_color_arr = render_pixel_jit(x, y, self.shape_params, self.shape_colors, background_color)
        
        # Convert the float array result back to a standard integer tuple
        return tuple(map(int, final_color_arr))