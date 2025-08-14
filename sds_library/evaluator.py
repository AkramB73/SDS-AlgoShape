import numpy as np
from typing import List, Tuple
import cv2

from .agent import Agent
from .shapes import Circle, Rectangle, Triangle

PixelCoords = List[Tuple[int, int]]
ImageArray = np.ndarray
RGBA = Tuple[int, int, int, int]


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

