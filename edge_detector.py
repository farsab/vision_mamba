import cv2
import numpy as np

def detect_edges(image_path: str, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Perform Canny edge detection on an input image.
    
    Args:
        image_path: Path to the input image file.
        low_threshold: Lower bound for hysteresis thresholding.
        high_threshold: Upper bound for hysteresis thresholding.
        
    Returns:
        A 2D numpy array of the same width/height containing the edge map.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    edges = cv2.Canny(img, low_threshold, high_threshold)
    return edges
