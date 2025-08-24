# sds_library/evaluator.py

import numpy as np
from typing import List, Tuple
import cv2
from numba import njit, prange

from .shapes import Shape, Circle, Rectangle, Triangle
# The 'Agent' import will be handled inside the function to avoid circular dependencies

# --- Numba JIT-Compiled Rendering Functions ---

# This function is unchanged. It's the fast engine for rendering a single pixel.
@njit(fastmath=True, cache=True)
def render_pixel_jit(x: int, y: int, shape_params: np.ndarray, shape_colors: np.ndarray, background_color: np.ndarray) -> np.ndarray:
    final_color = background_color.copy()
    
    for i in range(shape_params.shape[0]):
        params = shape_params[i]
        shape_type = int(params[0])
        
        is_inside = False
        if shape_type == 0: # Circle
            is_inside = (x - params[1])**2 + (y - params[2])**2 < params[3]
        elif shape_type == 1: # Rectangle
            is_inside = (params[1] <= x < params[3]) and (params[2] <= y < params[4])
        elif shape_type == 2: # Triangle
            p1_x, p1_y, p2_x, p2_y, p3_x, p3_y = params[1], params[2], params[3], params[4], params[5], params[6]
            d1 = (x - p3_x) * (p2_y - p3_y) - (p2_x - p3_x) * (y - p3_y)
            d2 = (x - p1_x) * (p3_y - p1_y) - (p3_x - p1_x) * (y - p1_y)
            c = (y - p1_y) * (p2_x - p1_x) - (x - p1_x) * (p2_y - p1_y)
            if not ((d1 < 0 or d2 < 0 or c < 0) and (d1 > 0 or d2 > 0 or c > 0)):
                 is_inside = True

        if is_inside:
            fg_color = shape_colors[i]
            bg_color = final_color
            fg_alpha = fg_color[3] / 255.0
            bg_alpha = bg_color[3] / 255.0
            final_color[:3] = fg_color[:3] * fg_alpha + bg_color[:3] * (1.0 - fg_alpha)
            final_color[3] = (fg_alpha + bg_alpha * (1.0 - fg_alpha)) * 255.0

    return final_color

# ▼▼▼ THIS IS THE NEW PARALLEL BLOCK EVALUATOR ▼▼▼
@njit(parallel=True, fastmath=True, cache=True)
def parallel_block_test_jit(blocks: np.ndarray, block_size: int, shape_params: np.ndarray, shape_colors: np.ndarray, target_image: np.ndarray) -> float:
    """
    Calculates the total error across a set of sample blocks in parallel.
    """
    total_block_error = 0.0
    background_color = np.array((255, 255, 255, 255), dtype=np.float32)

    # The outer loop is parallel, iterating over each BLOCK.
    for i in prange(blocks.shape[0]):
        start_x, start_y = blocks[i, 0], blocks[i, 1]
        
        block_error = 0.0
        # These inner loops iterate over the pixels *within* a block.
        for y_offset in range(block_size):
            for x_offset in range(block_size):
                x = start_x + x_offset
                y = start_y + y_offset
                
                predicted_color = render_pixel_jit(x, y, shape_params, shape_colors, background_color)
                actual_color = target_image[y, x]
                pixel_error = np.sum(np.abs(predicted_color - actual_color.astype(np.float32)))
                block_error += pixel_error
        
        total_block_error += block_error
        
    return total_block_error

# --- Main Evaluator Functions ---

# ▼▼▼ THE PARTIAL_TEST FUNCTION IS NOW A SIMPLE WRAPPER ▼▼▼
def partial_test(agent: 'Agent', target_image: np.ndarray, blocks: List[Tuple[int, int]], block_size: int) -> float:
    """
    Prepares data and calls the fast, parallel JIT function to evaluate an agent using blocks.
    """
    from .agent import Agent # Import here to prevent circular dependency

    if not blocks:
        return 0.0

    blocks_arr = np.array(blocks, dtype=np.int32)
    
    total_pixel_error = parallel_block_test_jit(blocks_arr, block_size, agent.shape_params, agent.shape_colors, target_image)
    
    num_pixels_checked = len(blocks) * block_size * block_size
    return total_pixel_error / num_pixels_checked if num_pixels_checked > 0 else 0.0


# full_fitness remains unchanged as it evaluates the entire image
def full_fitness(agent_shapes: List[Shape], target_image: np.ndarray, background_color: tuple) -> float:
    h, w, _ = target_image.shape
    canvas = np.full((h, w, 4), background_color, dtype=np.float32)

    for shape in agent_shapes:
        if isinstance(shape, Circle):
            fx1, fy1 = shape.center[0] - shape.radius, shape.center[1] - shape.radius
            fx2, fy2 = shape.center[0] + shape.radius, shape.center[1] + shape.radius
        elif isinstance(shape, Rectangle):
            fx1, fy1 = shape.top_left[0], shape.top_left[1]
            fx2, fy2 = shape.bottom_right[0], shape.bottom_right[1]
        elif isinstance(shape, Triangle):
            points = np.array([shape.p1, shape.p2, shape.p3])
            fx1, fy1 = points.min(axis=0)
            fx2, fy2 = points.max(axis=0)
        else:
            continue
        x1, y1 = max(0, int(fx1)), max(0, int(fy1))
        x2, y2 = min(w, int(np.ceil(fx2))), min(h, int(np.ceil(fy2)))
        if x1 >= x2 or y1 >= y2:
            continue
        box_w, box_h = x2 - x1, y2 - y1
        shape_layer = np.zeros((box_h, box_w, 4), dtype=np.float32)
        shape_color_bgra = tuple(map(int, (shape.color[2], shape.color[1], shape.color[0], shape.color[3])))
        if isinstance(shape, Circle):
            center_local = (int(shape.center[0] - x1), int(shape.center[1] - y1))
            cv2.circle(shape_layer, center_local, int(shape.radius), shape_color_bgra, -1)
        elif isinstance(shape, Rectangle):
            cv2.rectangle(shape_layer, (0, 0), (box_w, box_h), shape_color_bgra, -1)
        elif isinstance(shape, Triangle):
            points_local = np.array([(int(p[0] - x1), int(p[1] - y1)) for p in [shape.p1, shape.p2, shape.p3]], dtype=np.int32)
            cv2.fillPoly(shape_layer, [points_local], shape_color_bgra)
        canvas_region = canvas[y1:y2, x1:x2]
        alpha = shape_layer[:, :, 3:4] / 255.0
        canvas[y1:y2, x1:x2] = shape_layer * alpha + canvas_region * (1.0 - alpha)

    rendered_image_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)
    diff = np.subtract(target_image.astype(np.float32), rendered_image_uint8.astype(np.float32))
    rmse = np.sqrt(np.mean(np.square(diff)))
    return float(rmse)