import cv2
import numpy as np

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

def sample_pixels(width: int, height: int, n: int): 
    xs = np.random.randint(0, width, n)
    ys = np.random.randint(0, height, n)
    return list(zip(xs, ys))

def display_image(window_name: str, image_array: np.ndarray, delay: int = 1):
    # Convert from internal RGBA to BGRA for display with cv2.imshow
    display_img = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(delay)


