# examples/quickstart_ide.py

import numpy as np
import os
import cv2 # Needed for display windows and resizing

# --- Import from your SDS library ---
try:
    from sds_library.utils import load_image, save_image
    from sds_library.palette import generate_palette_kmeans
    from sds_library.diffusion import DiffusionSearch
    from sds_library.shapes import Circle, Rectangle, Triangle, Shape
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from sds_library.utils import load_image, save_image
    from sds_library.palette import generate_palette_kmeans
    from sds_library.diffusion import DiffusionSearch
    from sds_library.shapes import Circle, Rectangle, Triangle, Shape


# --- Configuration ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# --- EDIT THESE VALUES ---
INPUT_IMAGE_PATH = "/Users/akrambellala/Desktop/1.jpg" # <-- SET YOUR IMAGE PATH HERE
OUTPUT_FILENAME  = "output_best_agent.png"

# Resize the image for faster processing
PROCESSING_WIDTH = 250
PROCESSING_HEIGHT = 250

ITERATIONS = 2000
N_AGENTS = 10
SHAPES_PER_AGENT = 500
PALETTE_SIZE = 100
N_SAMPLES = 100 # The Numba optimization makes this very fast
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def render_full_agent(agent: 'Agent', sds_instance: 'DiffusionSearch') -> np.ndarray:
    """Helper function to render a full image of a single agent."""
    target_image = sds_instance.target
    h, w, _ = target_image.shape
    canvas = np.full((h, w, 4), sds_instance.background_color, dtype=np.float32)

    for shape in agent.shapes:
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
            
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    """
    Runs the full Stochastic Diffusion Search process.
    """
    try:
        print(f"Loading target image from: {INPUT_IMAGE_PATH}")
        original_image = load_image(INPUT_IMAGE_PATH)
    except FileNotFoundError:
        print(f"Error: The file at '{INPUT_IMAGE_PATH}' was not found.")
        return
    
    print(f"Resizing image to {PROCESSING_WIDTH}x{PROCESSING_HEIGHT} for faster computation...")
    target_image = cv2.resize(
        original_image,
        (PROCESSING_WIDTH, PROCESSING_HEIGHT),
        interpolation=cv2.INTER_AREA
    )
    
    print(f"Generating a {PALETTE_SIZE}-color palette...")
    palette = generate_palette_kmeans(target_image, PALETTE_SIZE)

    sds = DiffusionSearch(
        target=target_image,
        palette=palette,
        n_agents=N_AGENTS,
        shapes_per_agent=SHAPES_PER_AGENT,
        n_samples=N_SAMPLES
    )
    sds.run(iterations=ITERATIONS)

    all_final_agents = sds.get_best_agents(n=sds.n_agents)

    if not all_final_agents:
        print("SDS process finished, but no best agent could be determined.")
        return
        
    top_agent = all_final_agents[0]
    print("Rendering the best agent...")
    final_image = render_full_agent(top_agent, sds)
    
    print(f"💾 Saving best agent's image to {OUTPUT_FILENAME}")
    save_image(OUTPUT_FILENAME, final_image)
    print("\n✅ Process Complete.")


if __name__ == "__main__":
    main()