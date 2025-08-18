import cv2
import numpy as np
import random
from sklearn.cluster import KMeans
from typing import List, Tuple 

Palette = List[Tuple[int, int, int]]

def generate_palette_kmeans(image: np.ndarray, n_colors: int) -> list:
    if n_colors <= 0:   
        raise ValueError("Number of colors must be greater than 0.")
    
    # Cluster only on the RGB channels, ignoring alpha.
    pixels = image[:, :, :3].reshape(-1, 3)

    pixels = pixels.astype(np.float32)
    kmeans = KMeans(n_clusters=n_colors, random_state=0)

    kmeans.fit(pixels)

    # The cluster centers are the representative colors of the palette.
    palette_colors = kmeans.cluster_centers_.astype(np.uint8)

    return [tuple(color) for color in palette_colors]
