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
OUTPUT_FILENAME  = "output_overlapped.png"

# Resize the image for faster processing
PROCESSING_WIDTH = 250
PROCESSING_HEIGHT = 250

ITERATIONS = 1000
PALETTE_SIZE = 250
N_SAMPLES = 2 # Samples per sector (this is fast with Numba)

# --- SECTOR-BASED CONFIGURATION ---
# The image will be divided into a grid of sectors (e.g., 5x5 = 25 sectors)
N_SECTORS_X = 4
N_SECTORS_Y = 4
# Each sector will have its own small population of agents
AGENTS_PER_SECTOR = 1
# Each agent in a sector will manage this many shapes
SHAPES_PER_AGENT = 20

# --- NEW OVERLAP PARAMETER ---
# A value of 0.25 means sectors will overlap by 25% of their width/height.
# Good values are typically between 0.15 and 0.5.
SECTOR_OVERLAP = 0.7 # Needs addjustment based on number of agents.
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

def render_final_image(all_shapes: list, width: int, height: int, background_color: tuple) -> np.ndarray:
    """
    Renders a final image by drawing a list of shapes onto a canvas.
    This function is used at the end of the process to create the output file.
    """
    print(f"Rendering final image with {len(all_shapes)} total shapes...")
    canvas = np.full((height, width, 4), background_color, dtype=np.float32)

    for shape in all_shapes:
        # Determine the shape's bounding box
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
        x2, y2 = min(width, int(np.ceil(fx2))), min(height, int(np.ceil(fy2)))
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        box_w, box_h = x2 - x1, y2 - y1
        shape_layer = np.zeros((box_h, box_w, 4), dtype=np.float32)
        
        # Convert shape's RGBA color to BGRA for OpenCV
        shape_color_bgra = (shape.color[2], shape.color[1], shape.color[0], shape.color[3])

        # Draw the shape onto the small layer with relative coordinates
        if isinstance(shape, Circle):
            center_local = (int(shape.center[0] - x1), int(shape.center[1] - y1))
            cv2.circle(shape_layer, center_local, int(shape.radius), shape_color_bgra, -1)
        elif isinstance(shape, Rectangle):
            # For rectangles, the bounding box *is* the shape
            cv2.rectangle(shape_layer, (0, 0), (box_w, box_h), shape_color_bgra, -1)
        elif isinstance(shape, Triangle):
            points_local = np.array([
                (int(p[0] - x1), int(p[1] - y1)) for p in [shape.p1, shape.p2, shape.p3]
            ], dtype=np.int32)
            cv2.fillPoly(shape_layer, [points_local], shape_color_bgra)

        # Alpha blend the shape layer onto the main canvas
        canvas_region = canvas[y1:y2, x1:x2]
        alpha = shape_layer[:, :, 3:4] / 255.0
        canvas[y1:y2, x1:x2] = shape_layer * alpha + canvas_region * (1.0 - alpha)
            
    return np.clip(canvas, 0, 255).astype(np.uint8)


def main():
    """
    Runs the full sector-based Stochastic Diffusion Search process.
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

    # Pack all configuration parameters into a dictionary
    sds_config = {
        'n_agents_per_sector': AGENTS_PER_SECTOR,
        'shapes_per_agent': SHAPES_PER_AGENT,
        'n_samples': N_SAMPLES,
        'background_color': (255, 255, 255, 255)
    }

    # Initialize the main SDS manager with the sector layout and config
    sds = DiffusionSearch(
        target=target_image,
        palette=palette,
        n_sectors_x=N_SECTORS_X,
        n_sectors_y=N_SECTORS_Y,
        sector_overlap=SECTOR_OVERLAP, # Pass the new parameter here
        **sds_config
    )
    
    # Run the main evolution loop
    sds.run(iterations=ITERATIONS)

    # Collect the final shapes from the best agent of each sector
    final_shapes = sds.get_final_shapes()

    if not final_shapes:
        print("SDS process finished, but no shapes were generated.")
        return
    
    # Render the composite image from the collected shapes
    final_image = render_final_image(
        all_shapes=final_shapes,
        width=PROCESSING_WIDTH,
        height=PROCESSING_HEIGHT,
        background_color=sds.background_color
    )

    print(f"💾 Saving final composite image to {OUTPUT_FILENAME}")
    save_image(OUTPUT_FILENAME, final_image)
    print("\n✅ Process Complete.")


if __name__ == "__main__":
    main()