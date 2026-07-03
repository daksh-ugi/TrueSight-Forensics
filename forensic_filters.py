import cv2
import numpy as np
from preprocessing import decode_image_bytes

def compute_luminance_gradient(image_bytes: bytes) -> np.ndarray | None:
    """Computes the Sobel luminance gradient map of an image to expose edge inconsistencies."""
    try:
        img = decode_image_bytes(image_bytes)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        mag = np.sqrt(gx**2 + gy**2)
        mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        gradient_map = mag_normalized.astype(np.uint8)
        color_mapped = cv2.applyColorMap(gradient_map, cv2.COLORMAP_VIRIDIS)
        return color_mapped
    except Exception:
        return None
