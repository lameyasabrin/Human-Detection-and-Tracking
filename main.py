"""
Main Processing Pipeline
Integrates detection and tracking for complete video processing
"""

import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
import logging

from detector import HumanDetector, Detection
from tracker import MultiTracker, Track

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackingPipeline:
    """
    Complete pipeline for human detection and tracking.
    """
    
    def __init__(
        self,
        detector_model: str = 'yolov8n',
        tracker_method: str = 'deepsort',
        device: str = 'cuda',
        confidence: float = 0.5,
        config_path: Optional[str] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            detector_model: Detection model name
            tracker_method: Tracking algorithm
            device: Computing device
            confidence: Detection confidence threshold
            config_path: Path to configuration file
        """
        # Load config if provided
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Initialize detector
        self.detector = HumanDetector(
            model=detector_model,
            device=device,
            confidence=confidence
        )
        
        # Initialize tracker
        self.tracker = MultiTracker(
            method=tracker_method,
            max_age=self.config.get('max_age', 30),
            min_hits=self.config.get('min_hits', 3),
            iou_threshold=self.config.get('iou_threshold', 0.3)
        )
        
        # Analytics
        self.analytics = {
            'total_detections': 0,
            'unique_tracks': set(),
            'frames_processed': 0,
            'avg_fps': 0,
            'zone_crossings': {}
        }
        
        logger.info("Pipeline initialized successfully")
    
    @staticmethod
    def _load_config(config_path: str) -> Dict:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            elif config_path.endswith('.yaml'):
                import yaml
                return yaml.safe_load(f)
        return {}
    
    @staticmethod
    def _default_config() -> Dict:
        """Return default configuration"""
        return {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3,
            'visualize': True,
            'save_video': True,
            'draw_trajectory': True
        }
    
    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        visualize: bool = True,
        save_analytics: bool = True
    ) -> Dict:
        """
        Process video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            visualize: Whether to show visualization
            save_analytics: Whether to save analytics
            
        Returns:
            Dictionary with processing results
        """
        # Open video
        if input_path.isdigit():
            cap = cv2.VideoCapture(int(input_path))
        else:
            cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing loop
        frame_times = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                start_time = time.time()
                
                # Detect humans
                detections = self.detector.detect(frame)
                self.analytics['total_detections'] += len(detections)
                
                # Track humans
                tracks = self.tracker.update(detections, frame)
                
                # Update analytics
                for track in tracks:
                    self.analytics['unique_tracks'].add(track.track_id)
                
                # Visualize
                if visualize or output_path:
                    output = self.tracker.draw_tracks(
                        frame, tracks,
                        draw_trajectory=self.config.get('draw_trajectory', True)
                    )
                    
                    # Add info overlay
                    output = self._add_info_overlay(output, len(detections), len(tracks))
                    
                    if visualize:
                        cv2.imshow('Tracking Pipeline', output)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if writer:
                        writer.write(output)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    avg_time = np.mean(frame_times[-30:])
                    current_fps = 1.0 / avg_time if avg_time > 0 else 0
                    logger.info(f"Frame {frame_count}/{total_frames}, FPS: {current_fps:.1f}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if visualize:
                cv2.destroyAllWindows()
        
        # Compute final analytics
        self.analytics['frames_processed'] = frame_count
        self.analytics['avg_fps'] = 1.0 / np.mean(frame_times) if frame_times else 0
        self.analytics['unique_tracks'] = len(self.analytics['unique_tracks'])
        
        # Save analytics
        if save_analytics and output_path:
            analytics_path = Path(output_path).with_suffix('.json')
            with open(analytics_path, 'w') as f:
                json.dump(self.analytics, f, indent=2)
            logger.info(f"Analytics saved to {analytics_path}")
        
        return self.analytics
    
    def process_stream(
        self,
        stream_url: str,
        output_path: Optional[str] = None,
        duration: Optional[int] = None
    ):
        """
        Process live stream.
        
        Args:
            stream_url: Stream URL (RTSP, HTTP, etc.)
            output_path: Path to save output
            duration: Duration in seconds (None for continuous)
        """
        logger.info(f"Processing stream: {stream_url}")
        
        # Similar to process_video but for streams
        # Would need additional handling for reconnection, buffering, etc.
        self.process_video(stream_url, output_path, visualize=True)
    
    def _add_info_overlay(
        self,
        frame: np.ndarray,
        num_detections: int,
        num_tracks: int
    ) -> np.ndarray:
        """Add information overlay to frame"""
        output = frame.copy()
        
        # Create semi-transparent overlay
        overlay = output.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
        
        # Add text
        info_lines = [
            f"Detections: {num_detections}",
            f"Active Tracks: {num_tracks}",
            f"Total Tracks: {len(self.analytics['unique_tracks'])}"
        ]
        
        y_offset = 35
        for line in info_lines:
            cv2.putText(output, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        return output
    
    def reset(self):
        """Reset pipeline state"""
        self.tracker.reset()
        self.analytics = {
            'total_detections': 0,
            'unique_tracks': set(),
            'frames_processed': 0,
            'avg_fps': 0,
            'zone_crossings': {}
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Human Detection and Tracking')
    parser.add_argument('--input', required=True, help='Input video path or camera ID')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--model', default='yolov8n', help='Detection model')
    parser.add_argument('--tracker', default='deepsort', help='Tracking method')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--config', help='Config file path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrackingPipeline(
        detector_model=args.model,
        tracker_method=args.tracker,
        device=args.device,
        confidence=args.confidence,
        config_path=args.config
    )
    
    # Process video
    results = pipeline.process_video(
        input_path=args.input,
        output_path=args.output,
        visualize=not args.no_display
    )
    
    # Print results
    print("\n=== Processing Complete ===")
    print(f"Frames processed: {results['frames_processed']}")
    print(f"Average FPS: {results['avg_fps']:.2f}")
    print(f"Total detections: {results['total_detections']}")
    print(f"Unique tracks: {results['unique_tracks']}")


if __name__ == "__main__":
    main()
