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
    target_image: ImageArray,
    background_color: RGBA
) -> float:
    
    h, w, _ = target_image.shape

    canvas = np.full((h, w, 4), background_color, dtype=np.float32)
    
    for shape in agent.shapes:
        temp_layer = np.zeros((h, w, 4), dtype=np.float32)

        # Swap RGBA to BGRA
        shape_color_bgra = (shape.color[2], shape.color[1], shape.color[0], shape.color[3])

        if isinstance(shape, Circle):
            cv2.circle(temp_layer, shape.center, shape.radius, shape_color_bgra, -1)
        elif isinstance(shape, Rectangle):
            pt1 = shape.top_left
            pt2 = (pt1[0] + shape.size[0], pt1[1] + shape.size[1])
            cv2.rectangle(temp_layer, pt1, pt2, shape_color_bgra, -1)
        elif isinstance(shape, Triangle):
            points = np.array([shape.p1, shape.p2, shape.p3], dtype=np.int32)
            cv2.fillPoly(temp_layer, [points], shape_color_bgra)

        # Alpha blend the temporary layer onto the main canvas.
        alpha = temp_layer[:, :, 3] / 255.0
        alpha_mask = np.dstack((alpha, alpha, alpha, alpha))
        canvas = temp_layer * alpha_mask + canvas * (1.0 - alpha_mask) 

    rendered_image = np.clip(canvas, 0, 255).astype(np.uint8)

    diff = np.subtract(target_image.astype(np.float32), rendered_image.astype(np.float32))
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

