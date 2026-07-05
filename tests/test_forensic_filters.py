import numpy as np
from PIL import Image
from io import BytesIO
from forensic_filters import compute_luminance_gradient

def create_test_image(size=(100, 100)):
    img = Image.new("RGB", size, color=(128, 128, 128))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_compute_luminance_gradient_returns_correct_shape():
    img_bytes = create_test_image(size=(120, 80))
    grad = compute_luminance_gradient(img_bytes)
    assert grad is not None
    assert grad.shape[:2] == (80, 120)
    assert grad.ndim == 3
    assert grad.shape[2] == 3
    assert grad.dtype == np.uint8

def test_compute_luminance_gradient_returns_none_on_invalid_data():
    grad = compute_luminance_gradient(b"invalid image bytes")
    assert grad is None
