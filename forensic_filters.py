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

def compute_noise_residual(image_bytes: bytes) -> np.ndarray | None:
    """Computes the noise residual map of an image using a median filter subtraction."""
    try:
        img = decode_image_bytes(image_bytes)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(np.float32)
        
        filtered = cv2.medianBlur(gray, 3).astype(np.float32)
        residual = np.abs(gray_f - filtered)
        
        amplified = residual * 15.0
        clipped = np.clip(amplified, 0, 255).astype(np.uint8)
        
        color_mapped = cv2.applyColorMap(clipped, cv2.COLORMAP_MAGMA)
        return color_mapped
    except Exception:
        return None

