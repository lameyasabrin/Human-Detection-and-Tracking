# Human Detection and Tracking System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)

A real-time computer vision system for detecting and tracking humans in video streams using state-of-the-art deep learning models.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Models](#models) • [Demo](#demo) • [Contributing](#contributing)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Models](#models)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a robust human detection and tracking system using modern computer vision techniques. It combines object detection (YOLO, Faster R-CNN) with multi-object tracking algorithms (DeepSORT, ByteTrack) to provide accurate real-time tracking of multiple individuals in video streams.

### Use Cases

- Surveillance and security monitoring
- Crowd analytics and people counting
- Smart retail analytics
- Traffic monitoring and pedestrian safety
- Sports analytics
- Social distancing monitoring

## ✨ Features

- **Multiple Detection Models**: Support for YOLOv8, YOLOv5, Faster R-CNN, and SSD
- **Advanced Tracking**: DeepSORT and ByteTrack algorithms for robust multi-object tracking
- **Real-time Processing**: Optimized for real-time video stream processing
- **Multiple Input Sources**: Webcam, video files, IP cameras, RTSP streams
- **GPU Acceleration**: CUDA support for faster inference
- **Trajectory Visualization**: Visual representation of tracked paths
- **Zone-based Analytics**: Define custom zones for entry/exit counting
- **Export Capabilities**: Save results as video, images, or JSON data
- **RESTful API**: Easy integration with web applications
- **Docker Support**: Containerized deployment

## 🏗️ Architecture

```
┌─────────────────┐
│  Video Input    │
│ (Camera/File)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Detection    │
│   (YOLO/RCNN)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Tracking     │
│   (DeepSORT)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analytics     │
│  & Rendering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Output      │
│ (Video/API/DB)  │
└─────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB RAM minimum (16GB recommended)

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/human-detection-tracking.git
cd human-detection-tracking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_models.py
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t human-tracking .

# Run the container
docker run --gpus all -p 5000:5000 -v $(pwd)/data:/app/data human-tracking
```

### Option 3: Using Conda

```bash
# Create conda environment
conda create -n human-tracking python=3.9
conda activate human-tracking

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 🎬 Quick Start

### Basic Detection and Tracking

```python
from src.detector import HumanDetector
from src.tracker import MultiTracker
import cv2

# Initialize detector and tracker
detector = HumanDetector(model='yolov8', device='cuda')
tracker = MultiTracker(method='deepsort')

# Process video
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect humans
    detections = detector.detect(frame)
    
    # Track humans
    tracked_objects = tracker.update(detections, frame)
    
    # Visualize
    frame = tracker.draw_tracks(frame, tracked_objects)
    cv2.imshow('Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Command Line Usage

```bash
# Track humans in a video file
python main.py --input video.mp4 --output output.mp4 --model yolov8

# Use webcam
python main.py --input 0 --model yolov5 --tracker deepsort

# Process RTSP stream
python main.py --input rtsp://camera-ip:554/stream --save-trajectory

# Enable zone counting
python main.py --input video.mp4 --zones config/zones.json --analytics
```

## 📖 Usage

### Advanced Configuration

Create a configuration file `config.yaml`:

```yaml
detector:
  model: yolov8n
  confidence: 0.5
  device: cuda
  input_size: 640

tracker:
  method: deepsort
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3

analytics:
  enable_heatmap: true
  enable_counting: true
  save_trajectories: true
  zones:
    - name: entrance
      points: [[100, 100], [200, 100], [200, 200], [100, 200]]
    - name: exit
      points: [[400, 100], [500, 100], [500, 200], [400, 200]]

output:
  save_video: true
  save_json: true
  fps: 30
  codec: mp4v
```

Run with configuration:

```bash
python main.py --config config.yaml --input video.mp4
```

### Python API

```python
from src.pipeline import TrackingPipeline

# Initialize pipeline
pipeline = TrackingPipeline(
    detector_model='yolov8',
    tracker_method='bytetrack',
    config_path='config.yaml'
)

# Process video
results = pipeline.process_video(
    input_path='video.mp4',
    output_path='output.mp4',
    visualize=True,
    save_analytics=True
)

# Access analytics
print(f"Total people detected: {results['total_count']}")
print(f"Average speed: {results['avg_speed']}")
print(f"Zone crossings: {results['zone_crossings']}")
```

### REST API

Start the API server:

```bash
python api/server.py --port 5000
```

Upload and process video:

```bash
curl -X POST http://localhost:5000/api/track \
  -F "video=@video.mp4" \
  -F "config={\"model\":\"yolov8\",\"confidence\":0.5}"
```

## 🤖 Models

### Detection Models

| Model | Speed | mAP | GPU Memory | Best For |
|-------|-------|-----|------------|----------|
| YOLOv8n | ⚡⚡⚡ | 37.3 | 1GB | Real-time, edge devices |
| YOLOv8s | ⚡⚡ | 44.9 | 2GB | Balanced performance |
| YOLOv8m | ⚡ | 50.2 | 4GB | High accuracy |
| Faster R-CNN | ⚡ | 42.0 | 5GB | Precision-critical |

### Tracking Algorithms

- **DeepSORT**: Deep learning-based tracker with appearance features
- **ByteTrack**: High-performance tracker for crowded scenes
- **SORT**: Simple online real-time tracker (baseline)
- **StrongSORT**: Enhanced SORT with appearance features

## ⚙️ Configuration

### Environment Variables

```bash
export TORCH_HOME=/path/to/models
export CUDA_VISIBLE_DEVICES=0
export DETECTION_CONFIDENCE=0.5
```

### Model Selection

```python
# YOLOv8 variants
detector = HumanDetector(model='yolov8n')  # Nano
detector = HumanDetector(model='yolov8s')  # Small
detector = HumanDetector(model='yolov8m')  # Medium

# Other models
detector = HumanDetector(model='fasterrcnn')
detector = HumanDetector(model='ssd')
```

## 📊 Performance

Benchmark on NVIDIA RTX 3080 (1920x1080 video):

| Model + Tracker | FPS | Accuracy | GPU Memory |
|----------------|-----|----------|------------|
| YOLOv8n + DeepSORT | 45 | 92% | 2GB |
| YOLOv8s + DeepSORT | 35 | 95% | 3GB |
| YOLOv8m + ByteTrack | 28 | 96% | 5GB |

## 📁 Project Structure

```
human-detection-tracking/
├── src/
│   ├── detector.py          # Detection models
│   ├── tracker.py           # Tracking algorithms
│   ├── pipeline.py          # Main processing pipeline
│   ├── analytics.py         # Analytics and metrics
│   └── utils/
│       ├── visualization.py # Drawing utilities
│       ├── video_utils.py   # Video I/O
│       └── metrics.py       # Performance metrics
├── models/                  # Pre-trained model weights
├── data/                    # Sample videos and datasets
├── demo/                    # Demo scripts and notebooks
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── api/                     # REST API server
├── config/                  # Configuration files
├── scripts/                 # Utility scripts
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🧪 Testing

Run unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_detector.py

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

Full documentation is available at [docs/](./docs/):

- [API Reference](docs/api.md)
- [Model Training](docs/training.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- DeepSORT implementation
- ByteTrack algorithm
- OpenCV community
- PyTorch team

