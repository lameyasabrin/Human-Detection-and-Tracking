"""
REST API Server for Human Detection and Tracking
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import uuid
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import HumanDetector
from tracker import MultiTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Global detector and tracker instances
detector = None
tracker = None


def initialize_models(model='yolov8n', tracker_method='deepsort'):
    """Initialize detection and tracking models"""
    global detector, tracker
    
    try:
        detector = HumanDetector(model=model, confidence=0.5)
        tracker = MultiTracker(method=tracker_method)
        logger.info(f"Models initialized: {model}, {tracker_method}")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}")
        raise


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'tracker_loaded': tracker is not None
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'detectors': ['yolov8n', 'yolov8s', 'yolov8m', 'yolov5', 'fasterrcnn'],
        'trackers': ['sort', 'deepsort', 'bytetrack']
    })


@app.route('/api/track', methods=['POST'])
def track_video():
    """
    Process video for human tracking.
    
    Expects:
        - video: Video file
        - model: Detection model (optional)
        - tracker: Tracking method (optional)
        - confidence: Confidence threshold (optional)
    
    Returns:
        JSON with tracking results and output video path
    """
    try:
        # Check if video file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Get parameters
        model = request.form.get('model', 'yolov8n')
        tracker_method = request.form.get('tracker', 'deepsort')
        confidence = float(request.form.get('confidence', 0.5))
        
        # Save uploaded video
        video_id = str(uuid.uuid4())
        input_path = UPLOAD_FOLDER / f"{video_id}_{video_file.filename}"
        output_path = OUTPUT_FOLDER / f"{video_id}_output.mp4"
        
        video_file.save(str(input_path))
        logger.info(f"Video saved: {input_path}")
        
        # Initialize models if needed
        if detector is None or tracker is None:
            initialize_models(model, tracker_method)
        
        # Process video
        import cv2
        cap = cv2.VideoCapture(str(input_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Analytics
        total_detections = 0
        unique_tracks = set()
        frames_processed = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect and track
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            
            # Update analytics
            total_detections += len(detections)
            for track in tracks:
                unique_tracks.add(track.track_id)
            
            # Draw results
            output_frame = tracker.draw_tracks(frame, tracks)
            out.write(output_frame)
            
            frames_processed += 1
        
        cap.release()
        out.release()
        
        # Clean up input file
        input_path.unlink()
        
        # Return results
        return jsonify({
            'success': True,
            'video_id': video_id,
            'output_path': str(output_path.name),
            'analytics': {
                'frames_processed': frames_processed,
                'total_detections': total_detections,
                'unique_tracks': len(unique_tracks)
            }
        })
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed video"""
    try:
        file_path = OUTPUT_FOLDER / filename
        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(str(file_path), as_attachment=True)
    
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect', methods=['POST'])
def detect_image():
    """
    Detect humans in a single image.
    
    Expects:
        - image: Image file
        - confidence: Confidence threshold (optional)
    
    Returns:
        JSON with detection results
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get parameters
        confidence = float(request.form.get('confidence', 0.5))
        
        # Read image
        import cv2
        import numpy as np
        
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Initialize detector if needed
        if detector is None:
            initialize_models()
        
        # Detect humans
        detections = detector.detect(image)
        
        # Format results
        results = []
        for det in detections:
            results.append({
                'bbox': det.bbox,
                'confidence': float(det.confidence),
                'class': det.class_name
            })
        
        return jsonify({
            'success': True,
            'num_detections': len(detections),
            'detections': results
        })
    
    except Exception as e:
        logger.error(f"Error detecting in image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update detector and tracker configuration"""
    try:
        data = request.json
        
        model = data.get('model', 'yolov8n')
        tracker_method = data.get('tracker', 'deepsort')
        
        initialize_models(model, tracker_method)
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated'
        })
    
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 500MB'}), 413


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Human Tracking API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--model', default='yolov8n', help='Detection model')
    parser.add_argument('--tracker', default='deepsort', help='Tracking method')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Initialize models
    logger.info("Initializing models...")
    initialize_models(args.model, args.tracker)
    
    # Start server
    logger.info(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
