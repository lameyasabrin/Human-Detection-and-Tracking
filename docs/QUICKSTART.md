# Quick Start Guide

Get up and running with human detection and tracking in 5 minutes!

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/human-detection-tracking.git
cd human-detection-tracking
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Download Models

```bash
python scripts/download_models.py
```

## Basic Usage

### Command Line

#### Process a Video File

```bash
python main.py --input video.mp4 --output output.mp4 --model yolov8n
```

#### Use Webcam

```bash
python main.py --input 0 --model yolov8n
```

#### Process RTSP Stream

```bash
python main.py --input rtsp://camera-ip:554/stream --output stream_output.mp4
```

### Python API

```python
from src.detector import HumanDetector
from src.tracker import MultiTracker
import cv2

# Initialize
detector = HumanDetector(model='yolov8n', confidence=0.5)
tracker = MultiTracker(method='deepsort')

# Process video
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect and track
    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)
    
    # Visualize
    output = tracker.draw_tracks(frame, tracks)
    cv2.imshow('Tracking', output)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Using the Pipeline

```python
from src.pipeline import TrackingPipeline

# Initialize pipeline
pipeline = TrackingPipeline(
    detector_model='yolov8n',
    tracker_method='bytetrack',
    confidence=0.5
)

# Process video
results = pipeline.process_video(
    input_path='video.mp4',
    output_path='output.mp4',
    visualize=True
)

print(f"Processed {results['frames_processed']} frames")
print(f"Detected {results['unique_tracks']} unique people")
```

## Configuration

Create a custom configuration file:

```yaml
detector:
  model: yolov8s
  confidence: 0.6

tracker:
  method: bytetrack
  max_age: 30
```

Use it:

```bash
python main.py --input video.mp4 --config config.yaml
```

## Docker Usage

### Build and Run

```bash
# Build image
docker build -t human-tracking .

# Run container
docker run --gpus all -p 5000:5000 -v $(pwd)/data:/app/data human-tracking
```

### Using Docker Compose

```bash
docker-compose up -d
```

## API Server

### Start Server

```bash
python api/server.py --port 5000
```

### Upload Video for Processing

```bash
curl -X POST http://localhost:5000/api/track \
  -F "video=@video.mp4" \
  -F "model=yolov8n" \
  -F "confidence=0.5"
```

### Detect in Image

```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@person.jpg" \
  -F "confidence=0.5"
```

## Common Use Cases

### Real-time Webcam Tracking

```bash
python main.py --input 0 --model yolov8n --tracker bytetrack
```

### High-Accuracy Tracking

```bash
python main.py --input video.mp4 --model yolov8m --confidence 0.6
```

### Fast Processing

```bash
python main.py --input video.mp4 --model yolov8n --device cuda
```

## Troubleshooting

### CUDA Out of Memory

- Use a smaller model: `yolov8n` instead of `yolov8m`
- Reduce input resolution
- Process fewer frames

### Low FPS

- Use GPU: `--device cuda`
- Use smaller model
- Reduce confidence threshold
- Skip frames

### Poor Detection Quality

- Increase confidence threshold
- Use larger model (yolov8m)
- Adjust lighting/video quality

## Next Steps

- Read the full [README](README.md)
- Check [example notebooks](demo/)
- Explore [API documentation](docs/api.md)
- Join discussions on GitHub

## Getting Help

- GitHub Issues: Report bugs or request features
- Documentation: Check the `docs/` folder
- Examples: See `demo/` for notebooks

Happy tracking! 🎯
