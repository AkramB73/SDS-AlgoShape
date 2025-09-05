import cv2
import numpy as np
from typing import List, Tuple
from sds_library.shapes import Circle, Rectangle, Triangle, Shape

def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Error: The file at {path} was not found.")
    
    if len(image.shape) == 2: # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.shape[2] == 3: # BGR image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
    elif image.shape[2] == 4: # BGRA image
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        
    return image

def save_image(path: str, image_array: np.ndarray):
    save_img = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(path, save_img)
    print(f"Image successfully saved to {path}")

def sample_pixels(width: int, height: int, n: int) -> List[Tuple[int, int]]: 
    xs = np.random.randint(0, width, n)
    ys = np.random.randint(0, height, n)
    return list(zip(xs, ys))

def sample_pixel_blocks(width: int, height: int, n: int, block_size: int) -> List[Tuple[int, int]]:
    # Subtract block_size from the bounds to prevent blocks from going off-image
    xs = np.random.randint(0, width - block_size, n)
    ys = np.random.randint(0, height - block_size, n)
    return list(zip(xs, ys))

def display_image(window_name: str, image_array: np.ndarray, delay: int = 1):
    # Convert from internal RGBA to BGRA for display with cv2.imshow
    display_img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(delay)

def render_full_agent(agent: 'Agent', sds_instance: 'DiffusionSearch') -> np.ndarray:
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