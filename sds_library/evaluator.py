import numpy as np
from typing import List, Tuple
import cv2

from .agent import Agent
from .shapes import Circle, Rectangle, Triangle

PixelCoords = List[Tuple[int, int]]
ImageArray = np.ndarray
RGBA = Tuple[int, int, int, int]


def full_fitness(
    agent: Agent,
    target_image: np.ndarray,
    background_color: tuple
) -> float:
    
    h, w, _ = target_image.shape
    canvas = np.full((h, w, 4), background_color, dtype=np.float32)

    # Optimised Rendering with Bounding Boxes 
    for shape in agent.shapes:
        if isinstance(shape, Circle):
            x1 = max(0, shape.center[0] - shape.radius)
            y1 = max(0, shape.center[1] - shape.radius)
            x2 = min(w, shape.center[0] + shape.radius)
            y2 = min(h, shape.center[1] + shape.radius)
        elif isinstance(shape, Rectangle):
            x1 = max(0, shape.top_left[0])
            y1 = max(0, shape.top_left[1])
            x2 = min(w, shape.top_left[0] + shape.size[0])
            y2 = min(h, shape.top_left[1] + shape.size[1])
        elif isinstance(shape, Triangle):
            all_x = [shape.p1[0], shape.p2[0], shape.p3[0]]
            all_y = [shape.p1[1], shape.p2[1], shape.p3[1]]
            x1 = max(0, min(all_x))
            y1 = max(0, min(all_y))
            x2 = min(w, max(all_x))
            y2 = min(h, max(all_y))
        
        # Ensure the box has a valid area
        if x1 >= x2 or y1 >= y2:
            continue


        box_h, box_w = int(y2 - y1), int(x2 - x1)
        shape_layer = np.zeros((box_h, box_w, 4), dtype=np.float32)
        
        shape_color_bgra = (shape.color[2], shape.color[1], shape.color[0], shape.color[3])

        if isinstance(shape, Circle):
            center_local = (shape.center[0] - x1, shape.center[1] - y1)
            cv2.circle(shape_layer, center_local, shape.radius, shape_color_bgra, -1)
        elif isinstance(shape, Rectangle):
            pt1_local = (shape.top_left[0] - x1, shape.top_left[1] - y1)
            pt2_local = (pt1_local[0] + shape.size[0], pt1_local[1] + shape.size[1])
            cv2.rectangle(shape_layer, pt1_local, pt2_local, shape_color_bgra, -1)
        elif isinstance(shape, Triangle):
            points_local = np.array([
                (p[0] - x1, p[1] - y1) for p in [shape.p1, shape.p2, shape.p3]
            ], dtype=np.int32)
            cv2.fillPoly(shape_layer, [points_local], shape_color_bgra)

        canvas_region = canvas[int(y1):int(y2), int(x1):int(x2)]
        alpha = shape_layer[:, :, 3:4] / 255.0 # Keep dimensions for broadcasting
        
        canvas[int(y1):int(y2), int(x1):int(x2)] = shape_layer * alpha + canvas_region * (1.0 - alpha)

    rendered_image_uint8 = np.clip(canvas, 0, 255).astype(np.uint8)

    diff = np.subtract(target_image.astype(np.float32), rendered_image_uint8.astype(np.float32))
    squared_error = np.square(diff)
    mean_squared_error = np.mean(squared_error)
    rmse = np.sqrt(mean_squared_error)

    return float(rmse)


def partial_test(
    agent: Agent,
    target_image: ImageArray,
    samples: PixelCoords,
    background_color: RGBA,
    error_threshold: float
) -> bool:
    total_error = 0.0

    for x, y in samples:

        predicted_color = agent.render_pixel(x, y, background_color)

        actual_color = target_image[y, x]

        error = np.sum(
            np.abs(
                np.array(predicted_color, dtype=np.float32) -
                np.array(actual_color, dtype=np.float32)
            )
        )
        total_error += error

    average_error = total_error / len(samples)

    return average_error < error_threshold

