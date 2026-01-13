"""
Unit tests for detector module
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import HumanDetector, Detection


class TestDetection:
    """Test Detection dataclass"""
    
    def test_detection_creation(self):
        """Test creating a detection object"""
        det = Detection(
            bbox=(10, 20, 100, 200),
            confidence=0.95,
            class_id=0,
            class_name='person'
        )
        
        assert det.bbox == (10, 20, 100, 200)
        assert det.confidence == 0.95
        assert det.class_id == 0
        assert det.class_name == 'person'


class TestHumanDetector:
    """Test HumanDetector class"""
    
    @pytest.fixture
    def dummy_frame(self):
        """Create a dummy frame for testing"""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = HumanDetector(model='yolov8n', confidence=0.5)
        
        assert detector.model_name == 'yolov8n'
        assert detector.confidence_threshold == 0.5
        assert detector.model is not None
    
    def test_detector_device_selection(self):
        """Test device selection"""
        detector = HumanDetector(model='yolov8n', device='cpu')
        assert detector.device == 'cpu'
    
    def test_detect_returns_list(self, dummy_frame):
        """Test that detect returns a list"""
        detector = HumanDetector(model='yolov8n', confidence=0.5)
        detections = detector.detect(dummy_frame)
        
        assert isinstance(detections, list)
    
    def test_detect_with_empty_frame(self):
        """Test detection on empty frame"""
        detector = HumanDetector(model='yolov8n', confidence=0.5)
        empty_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(empty_frame)
        
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_visualize_detections(self, dummy_frame):
        """Test visualization of detections"""
        detector = HumanDetector(model='yolov8n', confidence=0.5)
        
        # Create dummy detection
        detections = [
            Detection(
                bbox=(50, 50, 150, 200),
                confidence=0.9,
                class_id=0,
                class_name='person'
            )
        ]
        
        output = detector.visualize(dummy_frame, detections)
        
        assert output.shape == dummy_frame.shape
        assert not np.array_equal(output, dummy_frame)  # Should be modified
    
    def test_info_property(self):
        """Test info property"""
        detector = HumanDetector(model='yolov8n', confidence=0.5)
        info = detector.info
        
        assert 'model' in info
        assert 'device' in info
        assert 'confidence_threshold' in info
        assert info['model'] == 'yolov8n'
        assert info['confidence_threshold'] == 0.5
    
    def test_invalid_model(self):
        """Test initialization with invalid model"""
        with pytest.raises(ValueError):
            HumanDetector(model='invalid_model')
    
    def test_confidence_threshold(self):
        """Test confidence threshold filtering"""
        detector1 = HumanDetector(model='yolov8n', confidence=0.3)
        detector2 = HumanDetector(model='yolov8n', confidence=0.8)
        
        assert detector1.confidence_threshold == 0.3
        assert detector2.confidence_threshold == 0.8


class TestDetectorIntegration:
    """Integration tests for detector"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image with a rectangle (simulating a person)"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        # Draw a rectangle to simulate a person
        cv2.rectangle(img, (200, 100), (400, 400), (255, 255, 255), -1)
        return img
    
    def test_end_to_end_detection(self, sample_image):
        """Test complete detection workflow"""
        detector = HumanDetector(model='yolov8n', confidence=0.5)
        
        # Detect
        detections = detector.detect(sample_image)
        
        # Visualize
        output = detector.visualize(sample_image, detections)
        
        assert isinstance(detections, list)
        assert output.shape == sample_image.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
