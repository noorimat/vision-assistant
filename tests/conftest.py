import pytest
import numpy as np
import cv2
from pathlib import Path

@pytest.fixture
def sample_frame():
    """Create a sample BGR frame (640x480)"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simple rectangle to simulate an object
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    return frame

@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    
    img_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return str(img_path)

@pytest.fixture
def mock_detections():
    """Mock YOLO detection results"""
    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = [xyxy]
            self.conf = [conf]
            self.cls = [cls]
    
    class MockBoxes:
        def __init__(self):
            self.boxes = [
                MockBox([100, 100, 300, 300], 0.85, 0),  # person - close
                MockBox([400, 400, 450, 450], 0.65, 56),  # chair - far
                MockBox([200, 200, 280, 280], 0.45, 0),   # person - below threshold
            ]
        
        def __iter__(self):
            return iter(self.boxes)
    
    class MockResults:
        def __init__(self):
            self.boxes = MockBoxes()
    
    return MockResults()
