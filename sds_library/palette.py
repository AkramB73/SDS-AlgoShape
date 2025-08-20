import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from typing import List, Tuple 

Palette = List[Tuple[int, int, int]]

def generate_palette_kmeans(image: np.ndarray, n_colors: int) -> list:
    if n_colors <= 0:
        raise ValueError("Number of colors must be greater than 0.")
    
    pixels = image[:, :, :3].reshape(-1, 3)
    pixels = pixels.astype(np.float32)
    
    # n_init='auto' is the modern default and avoids FutureWarning
    kmeans = KMeans(n_clusters=n_colors, random_state=0, n_init='auto')
    kmeans.fit(pixels)
    palette_colors = kmeans.cluster_centers_.astype(np.uint8)

    # --- FIX IS HERE ---
    # Use map(int, color) to ensure every value is a standard Python integer,
    # not a numpy.uint8. This is the root cause of the OpenCV errors.
    return [tuple(map(int, color)) for color in palette_colors]