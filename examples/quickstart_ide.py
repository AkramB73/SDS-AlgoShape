# examples/quickstart_ide.py

import numpy as np
import os
import cv2
from typing import List
import copy
import random

# --- Import from your SDS library ---
try:
    from sds_library.utils import load_image, save_image, sample_pixel_blocks
    from sds_library.palette import generate_palette_kmeans
    from sds_library.diffusion import DiffusionSearch
    from sds_library.agent import Agent
    from sds_library.shapes import Circle, Rectangle, Triangle, Shape
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from sds_library.utils import load_image, save_image, sample_pixel_blocks
    from sds_library.palette import generate_palette_kmeans
    from sds_library.diffusion import DiffusionSearch
    from sds_library.agent import Agent
    from sds_library.shapes import Circle, Rectangle, Triangle, Shape


# --- Configuration ---
INPUT_IMAGE_PATH = "/Users/akrambellala/Desktop/1.jpg"
OUTPUT_FILENAME  = "final_best_agent.png"

PROCESSING_WIDTH = 250
PROCESSING_HEIGHT = 250
PALETTE_SIZE = 250

ITERATIONS_PHASE_1 = 5000
N_AGENTS_PHASE_1 = 20
SHAPES_PER_AGENT_PHASE_1 = 50
N_SAMPLES_PHASE_1 = 1
BLOCK_SIZE_PHASE_1 = 249



def render_full_agent(agent: 'Agent', sds_instance: 'DiffusionSearch') -> np.ndarray:
    # This helper function remains unchanged
    target_image = sds_instance.target
    h, w, _ = target_image.shape
    canvas = np.full((h, w, 4), sds_instance.background_color, dtype=np.float32)
    for shape in agent.shapes:
        if isinstance(shape, Circle):
            fx1, fy1, fx2, fy2 = shape.center[0] - shape.radius, shape.center[1] - shape.radius, shape.center[0] + shape.radius, shape.center[1] + shape.radius
        elif isinstance(shape, Rectangle):
            fx1, fy1, fx2, fy2 = shape.top_left[0], shape.top_left[1], shape.bottom_right[0], shape.bottom_right[1]
        elif isinstance(shape, Triangle):
            points = np.array([shape.p1, shape.p2, shape.p3])
            fx1, fy1 = points.min(axis=0); fx2, fy2 = points.max(axis=0)
        else: continue
        x1, y1, x2, y2 = max(0, int(fx1)), max(0, int(fy1)), min(w, int(np.ceil(fx2))), min(h, int(np.ceil(fy2)))
        if x1 >= x2 or y1 >= y2: continue
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
    return np.clip(canvas, 0, 255).astype(np.uint8)

def main():
    target_image = cv2.resize(load_image(INPUT_IMAGE_PATH), (PROCESSING_WIDTH, PROCESSING_HEIGHT))
    average_color = tuple(map(int, np.mean(target_image, axis=(0, 1))))
    print(f"Using average image color as background: {average_color}")
    palette = generate_palette_kmeans(target_image, PALETTE_SIZE)

    sds_config_1 = {
        'n_agents': N_AGENTS_PHASE_1,
        'shapes_per_agent': SHAPES_PER_AGENT_PHASE_1,
        'n_samples': N_SAMPLES_PHASE_1,
        'block_size': BLOCK_SIZE_PHASE_1,
        'background_color': average_color
    }
    sds_phase_1 = DiffusionSearch(target=target_image, palette=palette, **sds_config_1)
    sds_phase_1.run(iterations=ITERATIONS_PHASE_1)
    final_best_agent = sds_phase_1.get_best_agents(n=1)[0]
    

    # --- SAVING THE FINAL IMAGE ---
    print("\n--- Rendering and Saving Final Agent ---")
    final_image = render_full_agent(final_best_agent, sds_phase_1)
    save_image(OUTPUT_FILENAME, final_image)
    print(f"\n✅ All updates complete! Image saved to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    main()