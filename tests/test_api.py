from fastapi.testclient import TestClient
import pytest

import api.main as api_main


def test_api_rejects_non_image_upload():
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.txt", b"data", "text/plain")}
    )

    assert response.status_code == 400


def test_api_returns_prediction(monkeypatch):
    provided_inputs = []

    def fake_predict(image_bytes, temperature=1.0):
        provided_inputs.append(image_bytes)
        return {
            "label": "Real",
            "confidence": 0.8,
            "raw": [0.8],
            "face_detected": True,
            "face_box": (10, 20, 30, 40),
        }

    monkeypatch.setattr(
        api_main,
        "predict_image",
        fake_predict,
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "Real",
        "confidence": 0.8,
        "raw_scores": [0.8],
        "face_detected": True,
        "face_box": [10, 20, 30, 40],
        "confidence_threshold": 0.7,
        "is_uncertain": False,
    }
    assert provided_inputs == [b"data"]


def test_api_returns_prediction_without_face_metadata(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {"label": "Fake", "confidence": 0.2, "raw": [0.2]},
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "verdict": "Fake",
        "confidence": 0.2,
        "raw_scores": [0.2],
        "face_detected": False,
        "face_box": None,
        "confidence_threshold": 0.7,
        "is_uncertain": True,
    }


def test_api_rejects_oversized_upload_before_prediction(monkeypatch):
    monkeypatch.setattr(api_main, "MAX_UPLOAD_SIZE_BYTES", 3)
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("prediction should not run for oversized input"),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 413


def test_async_detect_returns_task_id(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {
            "label": "Real",
            "confidence": 0.95,
            "raw": [0.95],
            "face_detected": True,
            "face_box": (1, 2, 3, 4),
        },
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect/async", files={"file": ("sample.png", b"data", "image/png")}
    )

    assert response.status_code == 202
    assert "task_id" in response.json()
    assert isinstance(response.json()["task_id"], str)


def test_task_status_returns_completed(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {
            "label": "Fake",
            "confidence": 0.15,
            "raw": [0.15],
            "face_detected": True,
            "face_box": (5, 6, 7, 8),
        },
    )
    client = TestClient(api_main.app)

    submit_response = client.post(
        "/api/detect/async", files={"file": ("sample.png", b"data", "image/png")}
    )
    task_id = submit_response.json()["task_id"]

    status_response = client.get(f"/api/task/{task_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"
    assert status_response.json()["verdict"] == "Fake"
    assert status_response.json()["confidence"] == 0.15
    assert status_response.json()["raw_scores"] == [0.15]
    assert status_response.json()["face_detected"] is True
    assert status_response.json()["face_box"] == [5, 6, 7, 8]


def test_task_not_found_returns_404():
    client = TestClient(api_main.app)

    response = client.get("/api/task/nonexistent")

    assert response.status_code == 404


def test_async_detect_rejects_non_image(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("prediction should not run for invalid async uploads"),
    )
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect/async", files={"file": ("sample.txt", b"data", "text/plain")}
    )

    assert response.status_code == 400


def test_rate_limit_handler_calculates_correct_retry_after():
    from fastapi import Request
    from slowapi.errors import RateLimitExceeded
    from slowapi.wrappers import Limit
    from limits import parse_many
    import time

    # Create a mock Request
    class MockApp:
        class state:
            pass

    class MockRequest:
        def __init__(self):
            self.app = MockApp()
            self.state = MockApp.state()

    req = MockRequest()

    # Test case 1: when no state/limiter is available, fallback to get_expiry
    limit_item = parse_many("5/minute")[0]
    limit_wrapper = Limit(limit_item, lambda: "ip", None, False, None, None, None, 1, False)
    exc = RateLimitExceeded(limit_wrapper)

    response = api_main._rate_limit_exceeded_handler(req, exc)
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "60"

    # Test case 2: when limiter and view_rate_limit are present
    from limits.strategies import MovingWindowRateLimiter
    from limits.storage import MemoryStorage

    storage = MemoryStorage()
    strategy = MovingWindowRateLimiter(storage)

    class MockLimiter:
        def __init__(self):
            self.limiter = strategy

    req.app.state.limiter = MockLimiter()

    # Add a limit hit to the strategy
    key = "127.0.0.1"
    strategy.hit(limit_item, key)

    req.state.view_rate_limit = (limit_item, [key])

    response = api_main._rate_limit_exceeded_handler(req, exc)
    assert response.status_code == 429
    retry_after = response.headers.get("Retry-After")
    assert retry_after is not None
    assert int(retry_after) > 0
    assert int(retry_after) <= 60


def test_api_key_verification_enforced(monkeypatch):
    # Set API_KEY in the module
    monkeypatch.setattr(api_main, "API_KEY", "secret-key")
    
    # Mock predict_image
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {"label": "Real", "confidence": 0.8, "raw": [0.8]},
    )
    
    client = TestClient(api_main.app)
    
    # 1. Test POST /api/detect
    # Missing key
    response = client.post("/api/detect", files={"file": ("sample.png", b"data", "image/png")})
    assert response.status_code == 401
    
    # Invalid key
    response = client.post(
        "/api/detect",
        headers={"X-API-Key": "wrong-key"},
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 401
    
    # Valid key
    response = client.post(
        "/api/detect",
        headers={"X-API-Key": "secret-key"},
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 200
    
    # 2. Test POST /api/detect/async
    # Missing key
    response = client.post("/api/detect/async", files={"file": ("sample.png", b"data", "image/png")})
    assert response.status_code == 401
    
    # Invalid key
    response = client.post(
        "/api/detect/async",
        headers={"X-API-Key": "wrong-key"},
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 401
    
    # Valid key
    response = client.post(
        "/api/detect/async",
        headers={"X-API-Key": "secret-key"},
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 202
    
    # 3. Test GET /api/task/{task_id}
    # Missing key
    response = client.get("/api/task/some-task-id")
    assert response.status_code == 401
    
    # Invalid key
    response = client.get("/api/task/some-task-id", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
    
    # Valid key (task doesn't exist, should return 404 instead of 401)
    response = client.get("/api/task/some-task-id", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 404


def test_rate_limiting_is_enforced(monkeypatch):
    # Disable API key auth
    monkeypatch.setattr(api_main, "API_KEY", "")
    
    # Mock predict_image
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {"label": "Real", "confidence": 0.8, "raw": [0.8]},
    )
    
    # Clear the limiter's storage before the test
    api_main.limiter.limiter.storage.reset()
    
    client = TestClient(api_main.app)
    
    # Send 10 requests, which is the default limit (10/minute)
    for _ in range(10):
        response = client.post(
            "/api/detect",
            files={"file": ("sample.png", b"data", "image/png")},
        )
        assert response.status_code == 200
        
    # The 11th request should exceed the limit and return 429
    response = client.post(
        "/api/detect",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
    
    # Clear the limiter storage again so it doesn't affect other tests
    api_main.limiter.limiter.storage.reset()


def test_async_endpoint_rejects_missing_api_key(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "secret-key")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("should not reach prediction without auth"),
    )
    client = TestClient(api_main.app)
    response = client.post(
        "/api/detect/async",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 401


def test_rate_limiting_is_enforced_async(monkeypatch):
    # Disable API key auth
    monkeypatch.setattr(api_main, "API_KEY", "")
    
    # Clear the limiter's storage before the test
    api_main.limiter.limiter.storage.reset()
    
    client = TestClient(api_main.app)
    
    # Send 10 requests to /api/detect/async
    for _ in range(10):
        response = client.post(
            "/api/detect/async",
            files={"file": ("sample.png", b"data", "image/png")},
        )
        assert response.status_code == 202
        
    # The 11th request should exceed the limit and return 429
    response = client.post(
        "/api/detect/async",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
    
    # Clear the limiter storage again
    api_main.limiter.limiter.storage.reset()

def test_api_rejects_negative_temperature(monkeypatch):
    """Test that negative temperature is rejected with 422."""
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("should not reach prediction with invalid temp"),
    )
    client = TestClient(api_main.app)
    
    response = client.post(
        "/api/detect?temperature=-5.0",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 422


def test_api_rejects_zero_temperature(monkeypatch):
    """Test that zero temperature is rejected with 422."""
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("should not reach prediction with invalid temp"),
    )
    client = TestClient(api_main.app)
    
    response = client.post(
        "/api/detect?temperature=0.0",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 422


def test_api_rejects_temperature_above_100(monkeypatch):
    """Test that temperature > 100 is rejected with 422."""
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("should not reach prediction with invalid temp"),
    )
    client = TestClient(api_main.app)
    
    response = client.post(
        "/api/detect?temperature=101.0",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 422 


def test_api_accepts_valid_temperature(monkeypatch):
    """Test that valid temperature (0 < temp <= 100) is accepted."""
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {
            "label": "Real",
            "confidence": 0.8,
            "raw": [0.8],
        },
    )
    client = TestClient(api_main.app)
    
    response = client.post(
        "/api/detect?temperature=1.5",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["verdict"] == "Real"


def test_async_rejects_invalid_temperature(monkeypatch):
    """Test that async endpoint rejects invalid temperature with 422."""
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: pytest.fail("should not reach prediction with invalid temp"),
    )
    client = TestClient(api_main.app)
    
    response = client.post(
        "/api/detect/async?temperature=-5.0",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 422


def test_async_accepts_valid_temperature(monkeypatch):
    """Test that async endpoint accepts valid temperature."""
    monkeypatch.setattr(api_main, "API_KEY", "")
    monkeypatch.setattr(
        api_main,
        "predict_image",
        lambda _bytes, **kwargs: {
            "label": "Fake",
            "confidence": 0.2,
            "raw": [0.2], 
            "face_detected": True,
            "face_box": (5, 6, 7, 8),
        },
    )
    client = TestClient(api_main.app)
    
    response = client.post(
        "/api/detect/async?temperature=2.5",
        files={"file": ("sample.png", b"data", "image/png")},
    )
    assert response.status_code == 202
    assert "task_id" in response.json() 


def test_api_rejects_corrupted_image_with_400(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "")
    
    def fake_predict(image_bytes, temperature=1.0):
        raise ValueError("The uploaded file appears to be corrupted or is not a valid image.")

    monkeypatch.setattr(api_main, "predict_image", fake_predict)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect", files={"file": ("sample.png", b"corrupted-bytes", "image/png")}
    )

    assert response.status_code == 400
    assert "corrupted" in response.json()["detail"]


def test_async_detect_rejects_corrupted_image_with_400(monkeypatch):
    monkeypatch.setattr(api_main, "API_KEY", "")
    
    def fake_predict(image_bytes, temperature=1.0):
        raise TypeError("image_input must be a file path, raw bytes, or numpy array.")

    monkeypatch.setattr(api_main, "predict_image", fake_predict)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/detect/async", files={"file": ("sample.png", b"corrupted-bytes", "image/png")}
    )

    assert response.status_code == 400
    assert "must be" in response.json()["detail"]


