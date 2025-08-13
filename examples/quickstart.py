import numpy as np
import os

try:
    from sds_library.utils import save_image,load_image
    from sds_library.palette import generate_palette_kmeans
    from sds_library.agent import Agent
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from sds_library.utils import save_image,load_image
    from sds_library.palette import generate_palette_kmeans
    from sds_library.agent import Agent

# --- Configuration ---
IMG_WIDTH = 500
IMG_HEIGHT = 500
SHAPES_PER_AGENT = 30
PALETTE_SIZE = 8
BACKGROUND_COLOR = (255, 255, 255, 255) # White
OUTPUT_FILENAME = "smoke_test_output.png"


def main():
    print("Starting test with your image...")

    try:
        target_image = load_image("/Users/akrambellala/Desktop/1.jpg")
    except FileNotFoundError:
        print("ERROR: Could not find your image file.")
        return

    print(f"Generating a {PALETTE_SIZE}-color palette from your image...")
    palette = generate_palette_kmeans(target_image, PALETTE_SIZE)

    print(f"Creating an agent with {SHAPES_PER_AGENT} shapes...")
    agent_instance = Agent(
        img_size=(IMG_WIDTH, IMG_HEIGHT),
        palette=palette,
        shapes_per_agent=SHAPES_PER_AGENT
    )

    print("Rendering agent...")
    output_canvas = np.full((IMG_HEIGHT, IMG_WIDTH, 4), BACKGROUND_COLOR, dtype=np.uint8)
    canvas_float = output_canvas.astype(np.float32)

    for shape in agent_instance.shapes:
        for y in range(IMG_HEIGHT):
            for x in range(IMG_WIDTH):
                if shape.contains(x, y):
                    fg_color = np.array(shape.color, dtype=np.float32)
                    bg_color = canvas_float[y, x]
                    fg_alpha = fg_color[3] / 255.0
                    canvas_float[y, x, :3] = (fg_color[:3] * fg_alpha) + (bg_color[:3] * (1.0 - fg_alpha))
                    bg_alpha = bg_color[3] / 255.0
                    new_alpha = (fg_alpha + bg_alpha * (1.0 - fg_alpha)) * 255.0
                    canvas_float[y, x, 3] = new_alpha

    output_canvas = canvas_float.astype(np.uint8)

    print(f"Saving output to '{OUTPUT_FILENAME}'...")
    save_image(OUTPUT_FILENAME, output_canvas)
    print("Test complete.")


if __name__ == "__main__":
    main()