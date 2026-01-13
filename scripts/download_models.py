"""
Script to download pre-trained models
"""

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_yolo_models():
    """Download YOLOv8 models"""
    try:
        from ultralytics import YOLO
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        model_names = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
        
        for model_name in model_names:
            logger.info(f"Downloading {model_name}...")
            try:
                model = YOLO(model_name)
                logger.info(f"✓ {model_name} downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
        
        logger.info("YOLOv8 models downloaded")
        
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")


def download_torchvision_models():
    """Download torchvision models"""
    try:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        
        logger.info("Downloading Faster R-CNN model...")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        logger.info("✓ Faster R-CNN downloaded successfully")
        
    except ImportError:
        logger.error("torchvision not installed. Run: pip install torchvision")
    except Exception as e:
        logger.error(f"Failed to download torchvision models: {e}")


def main():
    """Main entry point"""
    logger.info("Starting model download...")
    
    # Create models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Download models
    download_yolo_models()
    download_torchvision_models()
    
    logger.info("\nModel download complete!")
    logger.info(f"Models saved to: {models_dir.absolute()}")


if __name__ == '__main__':
    main()
