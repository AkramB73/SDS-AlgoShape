# sds_library/spatial_grid.py

import numpy as np
from typing import List, Tuple
from .shapes import Shape, Triangle # Add other shapes if you use them

class SpatialGrid:
    def __init__(self, width: int, height: int, grid_size: int = 20):
        """
        Initializes a spatial grid to partition shapes.

        Args:
            width (int): The width of the canvas.
            height (int): The height of the canvas.
            grid_size (int): The number of cells in each dimension (e.g., 20x20 grid).
        """
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cell_width = width / grid_size
        self.cell_height = height / grid_size
        
        # The grid itself: a list of lists, where each inner list will hold shape indices.
        self.grid: List[List[int]] = [[] for _ in range(grid_size * grid_size)]

    def clear(self):
        """Resets the grid by emptying all cells."""
        for cell in self.grid:
            cell.clear()

    def add_shapes(self, shapes: List[Shape]):
        """
        Populates the grid with a list of shapes.
        Each shape is added to every grid cell it overlaps.
        """
        self.clear()
        for i, shape in enumerate(shapes):
            # Calculate the bounding box of the shape
            if isinstance(shape, Triangle):
                points = np.array([shape.p1, shape.p2, shape.p3])
                min_x, min_y = points.min(axis=0)
                max_x, max_y = points.max(axis=0)
            # Add elif for Circle, Rectangle if you use them
            else:
                continue

            # Determine which grid cells the bounding box overlaps
            start_col = max(0, int(min_x / self.cell_width))
            end_col = min(self.grid_size - 1, int(max_x / self.cell_width))
            start_row = max(0, int(min_y / self.cell_height))
            end_row = min(self.grid_size - 1, int(max_y / self.cell_height))

            # Add the shape's index to all overlapped cells
            for row in range(start_row, end_row + 1):
                for col in range(start_col, end_col + 1):
                    cell_index = row * self.grid_size + col
                    self.grid[cell_index].append(i)

    def get_flattened_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the grid into a format suitable for the GPU.

        Returns:
            A tuple containing:
            - grid_indices (np.ndarray): A flat list of all shape indices, cell by cell.
            - cell_offsets (np.ndarray): An array where each element stores the starting
              position and count of indices for the corresponding grid cell.
        """
        grid_indices = []
        cell_offsets = np.zeros((self.grid_size * self.grid_size, 2), dtype=np.int32)
        
        current_offset = 0
        for i, cell in enumerate(self.grid):
            count = len(cell)
            cell_offsets[i] = [current_offset, count]
            grid_indices.extend(cell)
            current_offset += count
            
        return np.array(grid_indices, dtype=np.int32), cell_offsets