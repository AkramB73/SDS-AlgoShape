import numpy as np
import os
import cv2
from typing import List

try:
    from sds_library.utils import load_image, save_image, render_full_agent
    from sds_library.palette import generate_palette_kmeans
    from sds_library.diffusion import DiffusionSearch
    from sds_library.agent import Agent
    from sds_library.shapes import Circle, Rectangle, Triangle, Shape
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from sds_library.utils import load_image, save_image, render_full_agent
    from sds_library.palette import generate_palette_kmeans
    from sds_library.diffusion import DiffusionSearch
    from sds_library.agent import Agent
    from sds_library.shapes import Circle, Rectangle, Triangle, Shape


# --- Configuration ---
INPUT_IMAGE_PATH = "/path/to/your/image.jpg" # Change this path
OUTPUT_FILENAME  = "final_best_agent.png"

PROCESSING_WIDTH = 250
PROCESSING_HEIGHT = 250

ITERATIONS = 500
N_AGENTS = 20
SHAPES_PER_AGENT = 200
PALETTE_SIZE = 200 
N_SAMPLES = 1000 
BLOCK_SIZE = 5 
shapes_used = [Triangle] # You can change this to [Circle], [Rectangle], or [Triangle, Circle] etc.

def main():
    target_image = cv2.resize(load_image(INPUT_IMAGE_PATH), (PROCESSING_WIDTH, PROCESSING_HEIGHT))
    
    average_color = tuple(map(int, np.mean(target_image, axis=(0, 1))))
    print(f"Using average image color as background: {average_color}")
    
    palette = generate_palette_kmeans(target_image, PALETTE_SIZE)

    sds_config = {
        'n_agents': N_AGENTS,
        'shapes_per_agent': SHAPES_PER_AGENT,
        'n_samples': N_SAMPLES,
        'block_size': BLOCK_SIZE,
        'background_color': average_color,
        'shape_types': shapes_used
    }

    sds = DiffusionSearch(
        target=target_image, 
        palette=palette,
        **sds_config
    )
    
    sds.run(iterations=ITERATIONS)

    best_agents = sds.get_best_agents(n=1)

    if not best_agents:
        print("SDS process finished, but no best agent could be determined.")
        return
        
    top_agent = best_agents[0]

    print("\n--- Rendering and Saving Best Agent ---")
    
    final_image = render_full_agent(top_agent, sds)
    
    save_image(OUTPUT_FILENAME, final_image)

    print(f"\n All updates complete. Image saved to '{OUTPUT_FILENAME}'.")

if __name__ == "__main__":
    main()