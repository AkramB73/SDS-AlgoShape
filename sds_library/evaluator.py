# sds_library/evaluator.py

import numpy as np
from typing import List, Tuple
import cv2
from .agent import Agent
from .shapes import Circle, Rectangle, Triangle, Shape

def full_fitness(agent_shapes: List[Shape], target_image: np.ndarray, background_color: tuple) -> float:
    h, w, _ = target_image.shape
    canvas = np.full((h, w, 4), background_color, dtype=np.float32)

    for shape in agent_shapes:
        # Determine the shape's bounding box as floats first
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

        # Clamp the bounding box to the screen and convert to integers
        x1, y1 = max(0, int(fx1)), max(0, int(fy1))
        x2, y2 = min(w, int(np.ceil(fx2))), min(h, int(np.ceil(fy2)))
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        box_w, box_h = x2 - x1, y2 - y1
        shape_layer = np.zeros((box_h, box_w, 4), dtype=np.float32)
        shape_color_bgra = tuple(map(int, (shape.color[2], shape.color[1], shape.color[0], shape.color[3])))

        # Draw the shape onto the small layer with relative coordinates
        if isinstance(shape, Circle):
            center_local = (int(shape.center[0] - x1), int(shape.center[1] - y1))
            cv2.circle(shape_layer, center_local, int(shape.radius), shape_color_bgra, -1)
        elif isinstance(shape, Rectangle):
            cv2.rectangle(shape_layer, (0, 0), (box_w, box_h), shape_color_bgra, -1)
        elif isinstance(shape, Triangle):
            points_local = np.array([
                (int(p[0] - x1), int(p[1] - y1)) for p in [shape.p1, shape.p2, shape.p3]
            ], dtype=np.int32)
            cv2.fillPoly(shape_layer, [points_local], shape_color_bgra)

        # The canvas region and the shape layer are now guaranteed to have the same dimensions
        canvas_region = canvas[y1:y2, x1:x2]
        alpha = shape_layer[:, :, 3:4] / 255.0
        canvas[y1:y2, x1:x2] = shape_layer * alpha + canvas_region * (1.0 - alpha)

    rendered_image_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)
    diff = np.subtract(target_image.astype(np.float32), rendered_image_uint8.astype(np.float32))
    rmse = np.sqrt(np.mean(np.square(diff)))
    return float(rmse)

def partial_test(agent: Agent, target_image: np.ndarray, samples: List[Tuple[int, int]]) -> float:
    total_error = 0.0
    for x, y in samples:
        predicted_color = agent.render_pixel(x, y)
        actual_color = target_image[y, x]
        error = np.sum(np.abs(np.array(predicted_color, dtype=np.float32) - np.array(actual_color, dtype=np.float32)))
        total_error += error
    return total_error / len(samples) if samples else 0.0