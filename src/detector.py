"""
Human Detection Module
Supports multiple detection models: YOLOv8, YOLOv5, Faster R-CNN
"""

import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Detection result container"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class HumanDetector:
    """
    Human detection using various deep learning models.
    
    Attributes:
        model_name: Name of the detection model
        device: Computing device (cuda/cpu)
        confidence_threshold: Minimum confidence for detections
    """
    
    # Class ID for person in COCO dataset
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model: str = 'yolov8n',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        confidence: float = 0.5,
        input_size: int = 640
    ):
        """
        Initialize detector.
        
        Args:
            model: Model name (yolov8n, yolov8s, yolov8m, yolov5, fasterrcnn)
            device: Device for inference
            confidence: Confidence threshold
            input_size: Input image size
        """
        self.model_name = model
        self.device = device
        self.confidence_threshold = confidence
        self.input_size = input_size
        self.model = None
        
        logger.info(f"Initializing {model} detector on {device}")
        self._load_model()
        
    def _load_model(self):
        """Load the detection model"""
        try:
            if 'yolov8' in self.model_name:
                self._load_yolov8()
            elif 'yolov5' in self.model_name:
                self._load_yolov5()
            elif 'fasterrcnn' in self.model_name:
                self._load_fasterrcnn()
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
                
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_yolov8(self):
        """Load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'{self.model_name}.pt')
            self.model.to(self.device)
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
    
    def _load_yolov5(self):
        """Load YOLOv5 model"""
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.to(self.device)
            self.model.conf = self.confidence_threshold
        except Exception as e:
            logger.error(f"Failed to load YOLOv5: {e}")
            raise
    
    def _load_fasterrcnn(self):
        """Load Faster R-CNN model"""
        try:
            import torchvision
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            logger.error("torchvision not installed. Run: pip install torchvision")
            raise
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect humans in a frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        detections = []
        
        try:
            if 'yolov8' in self.model_name:
                detections = self._detect_yolov8(frame)
            elif 'yolov5' in self.model_name:
                detections = self._detect_yolov5(frame)
            elif 'fasterrcnn' in self.model_name:
                detections = self._detect_fasterrcnn(frame)
                
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            
        return detections
    
    def _detect_yolov8(self, frame: np.ndarray) -> List[Detection]:
        """Detect using YOLOv8"""
        results = self.model(frame, conf=self.confidence_threshold, classes=[0])
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls == self.PERSON_CLASS_ID and conf >= self.confidence_threshold:
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_id=cls,
                        class_name='person'
                    ))
        
        return detections
    
    def _detect_yolov5(self, frame: np.ndarray) -> List[Detection]:
        """Detect using YOLOv5"""
        results = self.model(frame)
        df = results.pandas().xyxy[0]
        
        detections = []
        for _, row in df.iterrows():
            if row['name'] == 'person' and row['confidence'] >= self.confidence_threshold:
                detections.append(Detection(
                    bbox=(int(row['xmin']), int(row['ymin']), 
                          int(row['xmax']), int(row['ymax'])),
                    confidence=float(row['confidence']),
                    class_id=0,
                    class_name='person'
                ))
        
        return detections
    
    def _detect_fasterrcnn(self, frame: np.ndarray) -> List[Detection]:
        """Detect using Faster R-CNN"""
        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        detections = []
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= self.confidence_threshold:  # 1 is person in COCO
                x1, y1, x2, y2 = box.astype(int)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score),
                    class_id=0,
                    class_name='person'
                ))
        
        return detections
    
    def visualize(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detections on frame.
        
        Args:
            frame: Input image
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return output
    
    @property
    def info(self) -> Dict:
        """Get detector information"""
        return {
            'model': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'input_size': self.input_size
        }


if __name__ == "__main__":
    # Example usage
    detector = HumanDetector(model='yolov8n', confidence=0.5)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect humans
        detections = detector.detect(frame)
        
        # Visualize
        output = detector.visualize(frame, detections)
        
        # Display
        cv2.imshow('Human Detection', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
