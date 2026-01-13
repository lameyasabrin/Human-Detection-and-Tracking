"""
Multi-Object Tracking Module
Implements DeepSORT, ByteTrack, and SORT algorithms
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Tracked object container"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int = 0
    hits: int = 0
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    lost: bool = False
    
    def __post_init__(self):
        """Initialize trajectory with center point"""
        center = self.get_center()
        self.trajectory.append(center)
    
    def get_center(self) -> Tuple[int, int]:
        """Get bounding box center"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.age += 1
        self.lost = False
        center = self.get_center()
        self.trajectory.append(center)


class MultiTracker:
    """
    Multi-object tracker supporting multiple algorithms.
    
    Supports:
        - SORT (Simple Online Realtime Tracker)
        - DeepSORT (Deep Learning SORT)
        - ByteTrack
    """
    
    def __init__(
        self,
        method: str = 'deepsort',
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize tracker.
        
        Args:
            method: Tracking method (sort, deepsort, bytetrack)
            max_age: Maximum frames to keep alive a track without detection
            min_hits: Minimum hits to confirm a track
            iou_threshold: IoU threshold for matching
        """
        self.method = method.lower()
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        logger.info(f"Initialized {method} tracker")
        
        # Load appearance model for DeepSORT
        if self.method == 'deepsort':
            self._load_appearance_model()
    
    def _load_appearance_model(self):
        """Load ReID model for DeepSORT"""
        try:
            # Placeholder for appearance model
            # In production, load a ReID model like OSNet
            logger.info("Appearance model loaded (placeholder)")
            self.appearance_model = None
        except Exception as e:
            logger.warning(f"Failed to load appearance model: {e}")
            self.appearance_model = None
    
    def update(self, detections: List, frame: Optional[np.ndarray] = None) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            frame: Current frame (optional, needed for DeepSORT)
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        if self.method == 'sort':
            return self._update_sort(detections)
        elif self.method == 'deepsort':
            return self._update_deepsort(detections, frame)
        elif self.method == 'bytetrack':
            return self._update_bytetrack(detections)
        else:
            raise ValueError(f"Unknown tracking method: {self.method}")
    
    def _update_sort(self, detections: List) -> List[Track]:
        """Update using SORT algorithm"""
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.age += 1
        
        # Match detections to tracks using IoU
        matched_tracks, unmatched_detections, unmatched_tracks = \
            self._associate_detections_to_tracks(detections, self.tracks)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            det = detections[det_idx]
            self.tracks[track_idx].update(det.bbox, det.confidence)
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].lost = True
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                confidence=det.confidence
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks 
            if t.age < self.max_age or not t.lost
        ]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def _update_deepsort(self, detections: List, frame: Optional[np.ndarray]) -> List[Track]:
        """Update using DeepSORT algorithm"""
        # For now, use SORT with placeholder for appearance features
        # In production, extract appearance features and use for matching
        return self._update_sort(detections)
    
    def _update_bytetrack(self, detections: List) -> List[Track]:
        """Update using ByteTrack algorithm"""
        # Split detections into high and low confidence
        high_conf = [d for d in detections if d.confidence > 0.6]
        low_conf = [d for d in detections if 0.1 < d.confidence <= 0.6]
        
        # First association with high confidence detections
        matched_tracks, unmatched_high, unmatched_tracks = \
            self._associate_detections_to_tracks(high_conf, self.tracks)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            det = high_conf[det_idx]
            self.tracks[track_idx].update(det.bbox, det.confidence)
        
        # Second association with low confidence detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        matched_tracks2, unmatched_low, unmatched_tracks2 = \
            self._associate_detections_to_tracks(low_conf, remaining_tracks)
        
        # Update tracks matched with low confidence
        for track_idx, det_idx in matched_tracks2:
            det = low_conf[det_idx]
            remaining_tracks[track_idx].update(det.bbox, det.confidence)
        
        # Create new tracks
        for det_idx in unmatched_high:
            det = high_conf[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                confidence=det.confidence
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Mark remaining unmatched as lost
        for track_idx in unmatched_tracks2:
            remaining_tracks[track_idx].lost = True
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks 
            if t.age < self.max_age or not t.lost
        ]
        
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def _associate_detections_to_tracks(
        self,
        detections: List,
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks using IoU.
        
        Returns:
            matched: List of (track_idx, detection_idx) pairs
            unmatched_detections: Indices of unmatched detections
            unmatched_tracks: Indices of unmatched tracks
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for t, track in enumerate(tracks):
            for d, det in enumerate(detections):
                iou_matrix[t, d] = self._compute_iou(track.bbox, det.bbox)
        
        # Hungarian algorithm for optimal assignment
        matched_indices = []
        
        # Greedy matching as simple alternative to Hungarian
        track_indices = list(range(len(tracks)))
        det_indices = list(range(len(detections)))
        
        while len(track_indices) > 0 and len(det_indices) > 0:
            # Find maximum IoU
            max_iou = 0
            max_t, max_d = -1, -1
            
            for t in track_indices:
                for d in det_indices:
                    if iou_matrix[t, d] > max_iou:
                        max_iou = iou_matrix[t, d]
                        max_t, max_d = t, d
            
            if max_iou < self.iou_threshold:
                break
            
            matched_indices.append((max_t, max_d))
            track_indices.remove(max_t)
            det_indices.remove(max_d)
        
        unmatched_tracks = track_indices
        unmatched_detections = det_indices
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    @staticmethod
    def _compute_iou(bbox1: Tuple[int, int, int, int], 
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track], 
                   draw_trajectory: bool = True) -> np.ndarray:
        """
        Draw tracks on frame.
        
        Args:
            frame: Input image
            tracks: List of tracks
            draw_trajectory: Whether to draw trajectory
            
        Returns:
            Frame with drawn tracks
        """
        output = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Color based on track ID
            color = self._get_color(track.track_id)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            label = f"ID: {track.track_id}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw trajectory
            if draw_trajectory and len(track.trajectory) > 1:
                points = np.array(list(track.trajectory), dtype=np.int32)
                cv2.polylines(output, [points], False, color, 2)
        
        return output
    
    @staticmethod
    def _get_color(track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def reset(self):
        """Reset tracker state"""
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0


if __name__ == "__main__":
    # Example usage
    from detector import HumanDetector
    
    detector = HumanDetector(model='yolov8n')
    tracker = MultiTracker(method='bytetrack')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect
        detections = detector.detect(frame)
        
        # Track
        tracks = tracker.update(detections, frame)
        
        # Visualize
        output = tracker.draw_tracks(frame, tracks)
        
        cv2.imshow('Tracking', output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
