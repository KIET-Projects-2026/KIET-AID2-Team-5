"""
Complete Traffic Monitoring System with Advanced Violation Detection
- Vehicle Detection and Tracking
- Speed Estimation and Violations
- Traffic Signal Detection (Red/Yellow/Green)
- Red Light Violations
- Distance-based Violations (Unsafe Following Distance)
- Lane Violations
- Stop-line Violations
- Wrong-lane Driving Detection
- Smooth Video Output Processing
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import time
import threading
import subprocess
import os
import logging
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import queue

# Third-party imports
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    os.system("pip install ultralytics")
    from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI(
    title="Complete Traffic Monitoring System",
    description="Advanced vehicle detection with comprehensive violation detection",
    version="4.0.0"
)

# CORS Configuration - Allow both local and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://trafficmonitoringsystem.onrender.com",
        "https://*.vercel.app",  # Your Vercel domain
        "https://*.vercel.app",  # Allow all Vercel preview deployments
        "*"  # Remove this in production for security
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== CONFIGURATION ====================

class Config:
    """Global configuration"""
    CONFIDENCE_THRESHOLD = 0.45
    NMS_THRESHOLD = 0.45
    MAX_TRACKS = 100
    FRAME_SKIP = 1
    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    MAX_STREAMS = 4
    
    # Traffic Light Detection (RGB color ranges in HSV)
    ENABLE_TRAFFIC_LIGHT_DETECTION = True
    RED_LOWER = np.array([0, 120, 70])
    RED_UPPER = np.array([10, 255, 255])
    RED_LOWER2 = np.array([170, 120, 70])  # Second red range
    RED_UPPER2 = np.array([180, 255, 255])
    YELLOW_LOWER = np.array([15, 100, 100])
    YELLOW_UPPER = np.array([35, 255, 255])
    GREEN_LOWER = np.array([40, 50, 50])
    GREEN_UPPER = np.array([90, 255, 255])
    
    # Traffic light detection thresholds
    TRAFFIC_LIGHT_MIN_AREA = 50
    TRAFFIC_LIGHT_MAX_AREA = 8000
    TRAFFIC_LIGHT_MIN_CIRCULARITY = 0.5
    
    # Zebra crossing detection - STRICT criteria to avoid false positives
    ENABLE_ZEBRA_DETECTION = True
    ZEBRA_MIN_STRIPES = 4  # Minimum stripes needed (real zebra crossings have 5-10+)
    ZEBRA_STRIPE_MIN_WIDTH = 80  # Zebra stripes are WIDE (typically 40-60cm = 80+ pixels)
    ZEBRA_STRIPE_MAX_WIDTH = 600  # Maximum width (shouldn't span entire frame)
    ZEBRA_STRIPE_MIN_HEIGHT = 8  # Minimum thickness of each stripe
    ZEBRA_STRIPE_MAX_HEIGHT = 50  # Maximum thickness (stripes are thin relative to width)
    ZEBRA_MIN_ASPECT_RATIO = 4.0  # Width/Height ratio (zebra stripes are very wide)
    ZEBRA_MAX_SPACING = 40  # Maximum gap between stripes (they're close together)
    ZEBRA_MIN_SPACING = 8  # Minimum gap (can't be touching)
    ZEBRA_SPACING_TOLERANCE = 0.4  # How consistent spacing must be (40%)
    ZEBRA_MIN_CROSSING_WIDTH = 150  # Minimum total width of crossing
    ZEBRA_MIN_ALIGNMENT = 0.7  # How aligned stripes must be horizontally (70%)
    
    # Road line detection
    ENABLE_ROAD_LINE_DETECTION = True
    ROAD_LINE_MIN_LENGTH = 50
    ROAD_LINE_WHITE_THRESHOLD = 200
    
    # Speed Detection
    SPEED_LIMIT_KMH = 60
    PIXELS_PER_METER = 8.0
    SPEED_CALCULATION_FRAMES = 25
    MOVEMENT_THRESHOLD = 5.0  # pixels
    
    # Distance-based violations
    SAFE_DISTANCE_METERS = 3.0
    UNSAFE_DISTANCE_PIXELS = SAFE_DISTANCE_METERS * PIXELS_PER_METER
    
    # Lane detection
    ENABLE_LANE_DETECTION = True
    NUM_LANES = 3
    LANE_WIDTH_PIXELS = 200
    
    # Stop line detection
    STOP_LINE_Y = 400  # Y coordinate of stop line
    STOP_LINE_THRESHOLD = 50  # pixels
    
    # Video output settings
    OUTPUT_FPS = 30
    VIDEO_CODEC = 'mp4v'
    VIDEO_QUALITY = 90
    
    # Directories
    DATA_DIR = Path("data")
    VIOLATIONS_DIR = DATA_DIR / "violations"
    LOGS_DIR = DATA_DIR / "logs"
    UPLOADS_DIR = DATA_DIR / "uploads"
    OUTPUT_DIR = DATA_DIR / "output"
    
    for dir_path in [DATA_DIR, VIOLATIONS_DIR, LOGS_DIR, UPLOADS_DIR, OUTPUT_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== YOUTUBE STREAM HANDLER ====================

def get_youtube_stream_url(youtube_url):
    """Extract actual stream URL from YouTube video"""
    try:
        logger.info(f"Extracting YouTube stream from: {youtube_url}")
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]',
            '-g',
            youtube_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        stream_url = result.stdout.strip()
        if stream_url:
            logger.info(f"Stream URL extracted successfully")
            return stream_url
        else:
            logger.error("No stream URL found")
    except Exception as e:
        logger.error(f"Failed to get YouTube stream: {e}")
    return None

# ==================== TRAFFIC LIGHT DETECTION ====================

class TrafficLightDetector:
    """Detect actual traffic lights in frame using computer vision"""
    
    def __init__(self):
        self.current_state = None  # None means no traffic light detected
        self.state_history = deque(maxlen=15)
        self.detected_lights = []  # List of detected traffic light positions
        self.last_detection_time = 0
        self.detection_confidence = 0.0
        self.state_change_time = time.time()
        self.is_detected = False  # Flag to indicate if traffic light is actually detected
        logger.info("Traffic Light Detector initialized - Real detection mode")
    
    def detect_traffic_lights(self, frame):
        """
        Detect traffic lights in frame using color and shape analysis.
        Returns list of detected lights with their states and positions.
        """
        self.detected_lights = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Search primarily in upper portion of frame (where signals usually are)
        search_height = int(frame.shape[0] * 0.6)
        search_region = hsv[:search_height, :]
        frame_region = frame[:search_height, :]
        
        # Detect each color
        colors_detected = []
        
        # Red detection
        red_mask1 = cv2.inRange(search_region, Config.RED_LOWER, Config.RED_UPPER)
        red_mask2 = cv2.inRange(search_region, Config.RED_LOWER2, Config.RED_UPPER2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_lights = self._find_light_blobs(red_mask, frame_region, "RED")
        colors_detected.extend(red_lights)
        
        # Yellow detection
        yellow_mask = cv2.inRange(search_region, Config.YELLOW_LOWER, Config.YELLOW_UPPER)
        yellow_lights = self._find_light_blobs(yellow_mask, frame_region, "YELLOW")
        colors_detected.extend(yellow_lights)
        
        # Green detection
        green_mask = cv2.inRange(search_region, Config.GREEN_LOWER, Config.GREEN_UPPER)
        green_lights = self._find_light_blobs(green_mask, frame_region, "GREEN")
        colors_detected.extend(green_lights)
        
        self.detected_lights = colors_detected
        self.is_detected = len(colors_detected) > 0
        
        return colors_detected
    
    def _find_light_blobs(self, mask, frame_region, color_name):
        """Find circular light blobs in a color mask"""
        detected = []
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if Config.TRAFFIC_LIGHT_MIN_AREA < area < Config.TRAFFIC_LIGHT_MAX_AREA:
                # Check circularity (traffic lights are typically circular)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity > Config.TRAFFIC_LIGHT_MIN_CIRCULARITY:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h if h > 0 else 0
                        
                        # Traffic light should be roughly circular
                        if 0.5 < aspect_ratio < 2.0:
                            # Calculate brightness in this region
                            roi = frame_region[y:y+h, x:x+w]
                            if roi.size > 0:
                                brightness = np.mean(roi)
                                
                                # Traffic lights should be bright
                                if brightness > 80:
                                    center = (x + w // 2, y + h // 2)
                                    detected.append({
                                        'color': color_name,
                                        'position': (x, y, x + w, y + h),
                                        'center': center,
                                        'area': area,
                                        'brightness': brightness,
                                        'confidence': min(circularity * brightness / 255, 1.0)
                                    })
        
        return detected
    
    def detect_traffic_light_state(self, frame):
        """Detect current traffic light state from frame"""
        current_time = time.time()
        
        # Detect all traffic lights
        lights = self.detect_traffic_lights(frame)
        
        if not lights:
            # No traffic light detected
            self.is_detected = False
            self.detection_confidence = 0.0
            return None, 0.0
        
        # Find the most prominent light (brightest and largest)
        best_light = max(lights, key=lambda x: x['brightness'] * x['area'])
        detected_state = best_light['color']
        confidence = best_light['confidence']
        
        # Add to history for stability
        self.state_history.append(detected_state)
        
        # Use majority voting for stable state
        if len(self.state_history) >= 5:
            from collections import Counter
            state_counts = Counter(self.state_history)
            most_common_state = state_counts.most_common(1)[0][0]
            
            if most_common_state != self.current_state:
                self.current_state = most_common_state
                self.state_change_time = current_time
                logger.info(f"Traffic light detected: {self.current_state}")
        else:
            self.current_state = detected_state
        
        self.detection_confidence = confidence
        self.last_detection_time = current_time
        self.is_detected = True
        
        return self.current_state, confidence
    
    def draw_traffic_light_info(self, frame):
        """Draw detected traffic lights on frame - only if actually detected"""
        if not self.is_detected or not self.detected_lights:
            return frame
        
        color_map = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0),
        }
        
        for light in self.detected_lights:
            x1, y1, x2, y2 = light['position']
            color = color_map.get(light['color'], (128, 128, 128))
            
            # Draw circle around detected light
            center = light['center']
            radius = max((x2 - x1), (y2 - y1)) // 2 + 5
            cv2.circle(frame, center, radius, color, 3)
            
            # Draw label
            label = f"{light['color']} ({light['confidence']*100:.0f}%)"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


# ==================== ZEBRA CROSSING DETECTION ====================

class ZebraCrossingDetector:
    """
    Detect zebra crossings (pedestrian crossings) with HIGH ACCURACY.
    
    Key characteristics of zebra crossings vs road lines:
    - Zebra stripes are WIDE (perpendicular to road direction)
    - Zebra stripes are closely and EVENLY spaced
    - Zebra stripes are roughly HORIZONTAL in the camera view
    - Multiple stripes (typically 5-10) form a crossing
    - All stripes have similar width and are aligned
    
    Road lane lines are:
    - NARROW and LONG (parallel to road direction)
    - Far apart from each other
    - Usually vertical/diagonal in camera view
    """
    
    def __init__(self):
        self.detected_crossings = []
        self.is_detected = False
        self.crossing_region = None
        self.detection_confidence = 0.0
        logger.info("Zebra Crossing Detector initialized - High accuracy mode")
    
    def detect_zebra_crossing(self, frame):
        """
        Detect zebra crossing with STRICT criteria.
        Only detects when confident it's a real zebra crossing.
        """
        self.detected_crossings = []
        self.is_detected = False
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on road area (middle to lower portion of frame)
        road_region_start = int(frame.shape[0] * 0.35)
        road_region_end = int(frame.shape[0] * 0.95)
        road_region = gray[road_region_start:road_region_end, :]
        
        # Apply adaptive thresholding for better white detection
        # This handles varying lighting conditions better
        blurred = cv2.GaussianBlur(road_region, (5, 5), 0)
        
        # Binary threshold for white regions
        _, white_mask = cv2.threshold(blurred, Config.ROAD_LINE_WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up and connect horizontal stripes
        # Use horizontal kernel to favor horizontal shapes (zebra stripes)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_h)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_h)
        
        # Find contours
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for zebra stripe candidates with STRICT criteria
        stripe_candidates = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip if too small or too large
            if w < Config.ZEBRA_STRIPE_MIN_WIDTH or w > Config.ZEBRA_STRIPE_MAX_WIDTH:
                continue
            if h < Config.ZEBRA_STRIPE_MIN_HEIGHT or h > Config.ZEBRA_STRIPE_MAX_HEIGHT:
                continue
            
            # Calculate aspect ratio (width / height)
            aspect_ratio = w / h if h > 0 else 0
            
            # Zebra stripes must be MUCH wider than tall
            if aspect_ratio < Config.ZEBRA_MIN_ASPECT_RATIO:
                continue
            
            # Check contour area vs bounding box area (should be filled)
            contour_area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0
            
            # Zebra stripes should be fairly solid (>50% filled)
            if fill_ratio < 0.5:
                continue
            
            # This looks like a potential zebra stripe
            stripe_candidates.append({
                'bbox': (x, y + road_region_start, x + w, y + h + road_region_start),
                'center_x': x + w // 2,
                'center_y': y + h // 2 + road_region_start,
                'width': w,
                'height': h,
                'aspect_ratio': aspect_ratio,
                'fill_ratio': fill_ratio
            })
        
        # Need minimum number of stripe candidates
        if len(stripe_candidates) < Config.ZEBRA_MIN_STRIPES:
            return self.detected_crossings
        
        # Sort candidates by Y position (top to bottom)
        stripe_candidates.sort(key=lambda s: s['center_y'])
        
        # Try to find groups of stripes that form a zebra crossing
        best_crossing = None
        best_score = 0
        
        for i in range(len(stripe_candidates) - Config.ZEBRA_MIN_STRIPES + 1):
            # Try different group sizes
            for group_size in range(Config.ZEBRA_MIN_STRIPES, min(len(stripe_candidates) - i + 1, 12)):
                group = stripe_candidates[i:i + group_size]
                
                # Validate this group as a potential zebra crossing
                is_valid, score, crossing_data = self._validate_zebra_group(group)
                
                if is_valid and score > best_score:
                    best_score = score
                    best_crossing = crossing_data
        
        if best_crossing:
            self.detected_crossings.append(best_crossing)
            self.is_detected = True
            self.crossing_region = best_crossing['bbox']
            self.detection_confidence = best_crossing['confidence']
        
        return self.detected_crossings
    
    def _validate_zebra_group(self, stripes):
        """
        Validate if a group of stripes forms a real zebra crossing.
        Returns (is_valid, score, crossing_data)
        """
        if len(stripes) < Config.ZEBRA_MIN_STRIPES:
            return False, 0, None
        
        # Check 1: Spacing between stripes should be consistent and small
        y_positions = [s['center_y'] for s in stripes]
        spacings = [y_positions[j+1] - y_positions[j] for j in range(len(y_positions)-1)]
        
        if not spacings:
            return False, 0, None
        
        avg_spacing = sum(spacings) / len(spacings)
        
        # Spacing should be within reasonable range for zebra crossings
        if avg_spacing < Config.ZEBRA_MIN_SPACING or avg_spacing > Config.ZEBRA_MAX_SPACING:
            return False, 0, None
        
        # Check spacing consistency
        spacing_variance = sum(abs(s - avg_spacing) for s in spacings) / len(spacings)
        spacing_consistency = 1 - (spacing_variance / avg_spacing) if avg_spacing > 0 else 0
        
        if spacing_consistency < (1 - Config.ZEBRA_SPACING_TOLERANCE):
            return False, 0, None
        
        # Check 2: Stripes should be horizontally aligned (similar X centers)
        x_centers = [s['center_x'] for s in stripes]
        avg_x = sum(x_centers) / len(x_centers)
        x_variance = sum(abs(x - avg_x) for x in x_centers) / len(x_centers)
        
        # Average width of stripes
        avg_width = sum(s['width'] for s in stripes) / len(stripes)
        
        # X alignment should be good (variance less than half the stripe width)
        alignment_score = max(0, 1 - (x_variance / (avg_width / 2))) if avg_width > 0 else 0
        
        if alignment_score < Config.ZEBRA_MIN_ALIGNMENT:
            return False, 0, None
        
        # Check 3: Stripes should have similar widths
        widths = [s['width'] for s in stripes]
        width_variance = sum(abs(w - avg_width) for w in widths) / len(widths)
        width_consistency = max(0, 1 - (width_variance / avg_width)) if avg_width > 0 else 0
        
        if width_consistency < 0.5:  # Widths should be at least 50% consistent
            return False, 0, None
        
        # Check 4: Total crossing width should be reasonable
        min_x = min(s['bbox'][0] for s in stripes)
        max_x = max(s['bbox'][2] for s in stripes)
        crossing_width = max_x - min_x
        
        if crossing_width < Config.ZEBRA_MIN_CROSSING_WIDTH:
            return False, 0, None
        
        # Calculate overall confidence score
        num_stripes_score = min(len(stripes) / 6, 1.0)  # More stripes = higher confidence
        
        confidence = (
            spacing_consistency * 0.3 +
            alignment_score * 0.25 +
            width_consistency * 0.2 +
            num_stripes_score * 0.25
        )
        
        # Require minimum confidence
        if confidence < 0.6:
            return False, 0, None
        
        # Build crossing data
        min_y = min(s['bbox'][1] for s in stripes)
        max_y = max(s['bbox'][3] for s in stripes)
        
        crossing_data = {
            'bbox': (min_x, min_y, max_x, max_y),
            'num_stripes': len(stripes),
            'confidence': confidence,
            'avg_spacing': avg_spacing,
            'avg_stripe_width': avg_width,
            'crossing_width': crossing_width
        }
        
        return True, confidence, crossing_data
    
    def draw_zebra_crossing(self, frame):
        """Draw detected zebra crossings on frame - only if actually detected"""
        if not self.is_detected or not self.detected_crossings:
            return frame
        
        for crossing in self.detected_crossings:
            x1, y1, x2, y2 = crossing['bbox']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            
            # Draw corner markers for better visibility
            corner_len = 20
            color = (0, 255, 255)  # Cyan
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 3)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 3)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 3)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 3)
            
            # Draw label
            label = f"ZEBRA CROSSING ({crossing['confidence']*100:.0f}%)"
            stripes_info = f"{crossing['num_stripes']} stripes detected"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 25), (x1 + w + 10, y1), (0, 255, 255), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, stripes_info, (x1 + 5, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        return frame

# ==================== LANE DETECTION ====================

class LaneDetector:
    """Detect actual road lane lines using computer vision"""
    
    def __init__(self, frame_width, frame_height=720, num_lanes=3):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_lanes = num_lanes
        self.lane_width = frame_width // num_lanes
        
        # Detected lane lines
        self.detected_lines = []
        self.is_detected = False
        
        # Fallback lane boundaries (used for vehicle lane assignment)
        self.lanes = []
        for i in range(num_lanes):
            left = i * self.lane_width
            right = (i + 1) * self.lane_width
            center = (left + right) // 2
            self.lanes.append({
                'id': i,
                'left': left,
                'right': right,
                'center': center
            })
        
        logger.info(f"Lane Detector initialized - Real detection mode")
    
    def detect_lane_lines(self, frame):
        """
        Detect white lane lines on road using Hough transform.
        Returns list of detected lines.
        """
        self.detected_lines = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on road area (lower portion of frame)
        road_start = int(frame.shape[0] * 0.35)
        road_region = gray[road_start:, :]
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(road_region, (5, 5), 0)
        
        # Threshold to find white lines
        _, white_mask = cv2.threshold(blurred, Config.ROAD_LINE_WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Edge detection
        edges = cv2.Canny(white_mask, 50, 150)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=Config.ROAD_LINE_MIN_LENGTH,
            maxLineGap=30
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Adjust y coordinates for full frame
                y1 += road_start
                y2 += road_start
                
                # Calculate line length
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Calculate angle
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Filter: keep lines that are mostly vertical (lane dividers)
                # or horizontal (stop lines, crosswalk edges)
                if length >= Config.ROAD_LINE_MIN_LENGTH:
                    line_type = "unknown"
                    if 60 < angle < 120:  # Mostly vertical (lane lines)
                        line_type = "lane_divider"
                    elif angle < 30 or angle > 150:  # Mostly horizontal
                        line_type = "stop_line"
                    
                    self.detected_lines.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'length': length,
                        'angle': angle,
                        'type': line_type
                    })
        
        self.is_detected = len(self.detected_lines) > 0
        return self.detected_lines
    
    def get_vehicle_lane(self, bbox):
        """Determine which lane a vehicle is in"""
        x1, y1, x2, y2 = bbox
        vehicle_center_x = (x1 + x2) / 2
        
        for lane in self.lanes:
            if lane['left'] <= vehicle_center_x < lane['right']:
                return lane['id']
        
        # Default to nearest lane
        return min(range(self.num_lanes), key=lambda i: abs(self.lanes[i]['center'] - vehicle_center_x))
    
    def check_wrong_lane(self, bbox, expected_lane_direction='right'):
        """Check if vehicle is in wrong lane"""
        lane_id = self.get_vehicle_lane(bbox)
        
        if expected_lane_direction == 'right' and lane_id == 0:
            return True
        
        return False
    
    def detect_stop_line(self, frame):
        """Detect horizontal stop lines on the road"""
        stop_lines = []
        
        if not self.detected_lines:
            self.detect_lane_lines(frame)
        
        for line in self.detected_lines:
            if line['type'] == 'stop_line':
                stop_lines.append(line)
        
        return stop_lines
    
    def draw_lanes(self, frame):
        """Draw detected lane lines on frame - only if actually detected"""
        if not self.is_detected or not self.detected_lines:
            return frame
        
        for line in self.detected_lines:
            x1, y1 = line['start']
            x2, y2 = line['end']
            
            # Color based on line type
            if line['type'] == 'lane_divider':
                color = (255, 255, 255)  # White for lane dividers
                thickness = 2
            elif line['type'] == 'stop_line':
                color = (0, 255, 255)  # Yellow for stop lines
                thickness = 3
            else:
                color = (200, 200, 200)  # Gray for unknown
                thickness = 1
            
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        
        # Add label if lines detected
        if self.detected_lines:
            lane_count = sum(1 for l in self.detected_lines if l['type'] == 'lane_divider')
            stop_count = sum(1 for l in self.detected_lines if l['type'] == 'stop_line')
            label = f"Detected: {lane_count} lane lines, {stop_count} stop lines"
            cv2.putText(frame, label, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

# ==================== VEHICLE TRACKING ====================

class VehicleTrack:
    """Track individual vehicle with comprehensive violation detection"""
    
    def __init__(self, track_id: int, bbox: tuple, timestamp: float, lane_detector: LaneDetector):
        self.track_id = track_id
        self.positions = deque(maxlen=Config.SPEED_CALCULATION_FRAMES)
        self.timestamps = deque(maxlen=Config.SPEED_CALCULATION_FRAMES)
        self.bboxes = deque(maxlen=Config.SPEED_CALCULATION_FRAMES)
        
        center = self.get_center(bbox)
        self.positions.append(center)
        self.timestamps.append(timestamp)
        self.bboxes.append(bbox)
        
        self.vehicle_class = None
        self.speed_kmh = 0.0
        self.lane_detector = lane_detector
        self.current_lane = lane_detector.get_vehicle_lane(bbox)
        self.lane_changes = []
        
        # Violation flags
        self.violations = set()
        self.violation_recorded = set()
        
        # Traffic signal violation
        self.signal_state_when_crossed = None
        self.crossed_stop_line = False
        self.position_when_red = None
        
    def get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def update(self, bbox: tuple, timestamp: float):
        """Update vehicle track"""
        center = self.get_center(bbox)
        self.positions.append(center)
        self.timestamps.append(timestamp)
        self.bboxes.append(bbox)
        
        # Update lane
        new_lane = self.lane_detector.get_vehicle_lane(bbox)
        if new_lane != self.current_lane:
            self.lane_changes.append({
                'from': self.current_lane,
                'to': new_lane,
                'timestamp': timestamp
            })
            self.current_lane = new_lane
        
        # Calculate speed
        if len(self.positions) >= 2:
            self.calculate_speed()
    
    def calculate_speed(self):
        """Calculate vehicle speed in km/h"""
        if len(self.positions) < 2:
            return
        
        # Calculate total distance
        total_distance_pixels = 0
        for i in range(1, len(self.positions)):
            x1, y1 = self.positions[i-1]
            x2, y2 = self.positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance_pixels += distance
        
        # Calculate time elapsed
        time_elapsed_seconds = self.timestamps[-1] - self.timestamps[0]
        
        if time_elapsed_seconds > 0:
            distance_meters = total_distance_pixels / Config.PIXELS_PER_METER
            speed_ms = distance_meters / time_elapsed_seconds
            self.speed_kmh = speed_ms * 3.6
    
    def check_speed_violation(self):
        """Check if vehicle exceeds speed limit"""
        if self.speed_kmh > Config.SPEED_LIMIT_KMH:
            if 'speed' not in self.violations:
                self.violations.add('speed')
                return True
        return False
    
    def check_red_light_violation(self, signal_state: str):
        """Check if vehicle moves during red light"""
        if signal_state == "RED":
            if len(self.positions) >= 2:
                # Check if vehicle moved significantly
                x1, y1 = self.positions[-2]
                x2, y2 = self.positions[-1]
                movement = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                # Check if vehicle crossed stop line during red
                current_y = y2
                if current_y > Config.STOP_LINE_Y and movement > Config.MOVEMENT_THRESHOLD:
                    if not self.crossed_stop_line:
                        self.crossed_stop_line = True
                        self.signal_state_when_crossed = "RED"
                        
                        if 'red_light' not in self.violations:
                            self.violations.add('red_light')
                            return True
        else:
            # Reset when signal is not red
            if signal_state == "GREEN":
                self.crossed_stop_line = False
        
        return False
    
    def check_stop_line_violation(self, signal_state: str):
        """Check if vehicle crossed stop line when it shouldn't"""
        if signal_state in ["RED", "YELLOW"]:
            if len(self.bboxes) > 0:
                x1, y1, x2, y2 = self.bboxes[-1]
                vehicle_front_y = y2  # Bottom of bbox is front of vehicle
                
                # Check if vehicle is beyond stop line
                if vehicle_front_y > Config.STOP_LINE_Y + Config.STOP_LINE_THRESHOLD:
                    if 'stop_line' not in self.violations and signal_state == "RED":
                        self.violations.add('stop_line')
                        return True
        
        return False
    
    def check_lane_violation(self):
        """Check for improper lane changes"""
        if len(self.lane_changes) > 2:  # Frequent lane changes
            if 'lane_change' not in self.violations:
                self.violations.add('lane_change')
                return True
        return False
    
    def check_wrong_lane(self):
        """Check if vehicle is in wrong lane"""
        if self.lane_detector.check_wrong_lane(self.bboxes[-1] if self.bboxes else (0, 0, 0, 0)):
            if 'wrong_lane' not in self.violations:
                self.violations.add('wrong_lane')
                return True
        return False
    
    def check_unsafe_distance(self, other_track):
        """Check if following distance is unsafe"""
        if len(self.positions) == 0 or len(other_track.positions) == 0:
            return False
        
        # Calculate distance between vehicles
        my_pos = self.positions[-1]
        other_pos = other_track.positions[-1]
        
        distance = np.sqrt((my_pos[0] - other_pos[0])**2 + (my_pos[1] - other_pos[1])**2)
        
        # Check if in same lane and too close
        if self.current_lane == other_track.current_lane:
            if distance < Config.UNSAFE_DISTANCE_PIXELS:
                if 'unsafe_distance' not in self.violations:
                    self.violations.add('unsafe_distance')
                    return True
        
        return False

# ==================== SINGLE STREAM MONITOR ====================

class SingleStreamMonitor:
    """Monitor a single video stream with comprehensive violation detection"""
    
    def __init__(self, stream_id: int):
        self.stream_id = stream_id
        self.vehicle_detector = YOLO('yolov8n.pt')
        self.traffic_light_detector = TrafficLightDetector()
        self.zebra_crossing_detector = ZebraCrossingDetector()
        self.lane_detector = None  # Initialize when frame size is known
        
        self.active_tracks: Dict[int, VehicleTrack] = {}
        self.vehicle_count = defaultdict(int)
        self.violations = []
        self.current_frame = None
        self.processing = False
        self.stream_url = None
        
        # Statistics
        self.fps = 0
        self.detection_time = 0
        self.total_vehicles = 0
        self.frame_count = 0
        self.average_speed = 0.0
        self.current_signal_state = None  # None means no signal detected
        self.signal_confidence = 0.0
        
        # Violation counters
        self.violation_counts = {
            'speed': 0,
            'red_light': 0,
            'stop_line': 0,
            'lane_change': 0,
            'wrong_lane': 0,
            'unsafe_distance': 0
        }
        
        # Frame buffering for smooth output
        self.frame_buffer = deque(maxlen=30)
        self.buffer_lock = threading.Lock()
        
        # Video writer for output
        self.video_writer = None
        self.output_path = None
        
        logger.info(f"Stream Monitor {stream_id} initialized with real detection")
    
    def initialize_lane_detector(self, frame_width, frame_height):
        """Initialize lane detector with frame dimensions"""
        if self.lane_detector is None:
            self.lane_detector = LaneDetector(frame_width, frame_height, Config.NUM_LANES)
    
    def process_frame(self, frame):
        """Process a single frame with all detection algorithms"""
        start_time = time.time()
        
        # Resize frame
        frame = cv2.resize(frame, (Config.INPUT_WIDTH, Config.INPUT_HEIGHT))
        
        # Initialize lane detector
        if self.lane_detector is None:
            self.initialize_lane_detector(frame.shape[1], frame.shape[0])
        
        # Detect traffic light state (only updates if actually detected)
        if Config.ENABLE_TRAFFIC_LIGHT_DETECTION:
            state, confidence = self.traffic_light_detector.detect_traffic_light_state(frame)
            if state is not None:  # Only update if traffic light actually detected
                self.current_signal_state = state
                self.signal_confidence = confidence
        
        # Detect lane lines
        if Config.ENABLE_ROAD_LINE_DETECTION and self.lane_detector:
            self.lane_detector.detect_lane_lines(frame)
        
        # Detect zebra crossings
        if Config.ENABLE_ZEBRA_DETECTION:
            self.zebra_crossing_detector.detect_zebra_crossing(frame)
        
        # Detect and track vehicles
        current_time = time.time()
        vehicles = self.detect_and_track_vehicles(frame, current_time)
        
        # Check all violations
        self.check_all_violations(frame, current_time)
        
        # Draw all annotations
        annotated_frame = self.draw_annotations(frame, vehicles)
        
        # Buffer frame
        with self.buffer_lock:
            self.frame_buffer.append(annotated_frame.copy())
            self.current_frame = annotated_frame
        
        # Write to video file
        if self.video_writer is not None:
            self.video_writer.write(annotated_frame)
        
        self.detection_time = (time.time() - start_time) * 1000
        return annotated_frame
    
    def detect_and_track_vehicles(self, frame, current_time):
        """Detect and track vehicles using YOLO"""
        results = self.vehicle_detector.track(
            frame,
            conf=Config.CONFIDENCE_THRESHOLD,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        
        detections = []
        
        for result in results:
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    class_name = self.vehicle_detector.names[cls]
                    
                    # Filter for vehicles only
                    if class_name in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']:
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Update or create track
                        if track_id not in self.active_tracks:
                            self.active_tracks[track_id] = VehicleTrack(
                                track_id, (x1, y1, x2, y2), current_time, self.lane_detector
                            )
                            self.total_vehicles += 1
                            self.vehicle_count[class_name] += 1
                        else:
                            self.active_tracks[track_id].update((x1, y1, x2, y2), current_time)
                        
                        track = self.active_tracks[track_id]
                        track.vehicle_class = class_name
                        
                        detections.append({
                            'track_id': track_id,
                            'bbox': (x1, y1, x2, y2),
                            'class': class_name,
                            'confidence': float(conf),
                            'speed': track.speed_kmh,
                            'lane': track.current_lane,
                            'violations': list(track.violations)
                        })
        
        # Calculate average speed
        speeds = [track.speed_kmh for track in self.active_tracks.values() if track.speed_kmh > 0]
        if speeds:
            self.average_speed = sum(speeds) / len(speeds)
        
        # Cleanup old tracks
        self.cleanup_old_tracks(current_time)
        
        return detections
    
    def check_all_violations(self, frame, current_time):
        """Check all types of violations for all vehicles"""
        tracks_list = list(self.active_tracks.values())
        
        # Check if traffic light is actually detected
        signal_detected = self.traffic_light_detector.is_detected
        
        for track in tracks_list:
            # Speed violation (always check)
            if track.check_speed_violation():
                if 'speed' not in track.violation_recorded:
                    self.record_violation(frame, track, 'speed')
                    track.violation_recorded.add('speed')
            
            # Red light violation (only if signal is detected)
            if signal_detected and self.current_signal_state:
                if track.check_red_light_violation(self.current_signal_state):
                    if 'red_light' not in track.violation_recorded:
                        self.record_violation(frame, track, 'red_light')
                        track.violation_recorded.add('red_light')
            
            # Stop line violation (only if signal is detected and there's a detected stop line)
            if signal_detected and self.current_signal_state:
                # Also check if we detected any stop lines
                stop_lines = self.lane_detector.detect_stop_line(frame) if self.lane_detector else []
                if stop_lines or self.zebra_crossing_detector.is_detected:
                    if track.check_stop_line_violation(self.current_signal_state):
                        if 'stop_line' not in track.violation_recorded:
                            self.record_violation(frame, track, 'stop_line')
                            track.violation_recorded.add('stop_line')
            
            # Lane violation (check if lanes detected)
            if self.lane_detector and self.lane_detector.is_detected:
                if track.check_lane_violation():
                    if 'lane_change' not in track.violation_recorded:
                        self.record_violation(frame, track, 'lane_change')
                        track.violation_recorded.add('lane_change')
                
                # Wrong lane
                if track.check_wrong_lane():
                    if 'wrong_lane' not in track.violation_recorded:
                        self.record_violation(frame, track, 'wrong_lane')
                        track.violation_recorded.add('wrong_lane')
            
            # Unsafe distance (check against other vehicles - always active)
            for other_track in tracks_list:
                if other_track.track_id != track.track_id:
                    if track.check_unsafe_distance(other_track):
                        if 'unsafe_distance' not in track.violation_recorded:
                            self.record_violation(frame, track, 'unsafe_distance')
                            track.violation_recorded.add('unsafe_distance')
                        break
    
    def record_violation(self, frame, track: VehicleTrack, violation_type: str):
        """Record a violation"""
        self.violation_counts[violation_type] += 1
        
        # Convert numpy types to native Python types for JSON serialization
        violation_data = {
            'id': int(len(self.violations) + 1),
            'stream_id': int(self.stream_id),
            'track_id': int(track.track_id),
            'vehicle_class': str(track.vehicle_class) if track.vehicle_class else None,
            'speed_kmh': float(round(track.speed_kmh, 1)),
            'lane': int(track.current_lane),
            'violation_type': str(violation_type),
            'signal_state': str(self.current_signal_state),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add specific violation details
        if violation_type == 'speed':
            violation_data['speed_limit'] = int(Config.SPEED_LIMIT_KMH)
            violation_data['excess_speed'] = float(round(track.speed_kmh - Config.SPEED_LIMIT_KMH, 1))
        
        # Save violation image
        try:
            timestamp_str = int(time.time())
            violation_file = Config.VIOLATIONS_DIR / f"stream{self.stream_id}_{violation_type}_{violation_data['id']}_{timestamp_str}.jpg"
            cv2.imwrite(str(violation_file), frame)
            violation_data['image_path'] = str(violation_file)
        except Exception as e:
            logger.error(f"Failed to save violation image: {e}")
        
        self.violations.append(violation_data)
        
        logger.warning(f"Stream {self.stream_id} - {violation_type.upper()} VIOLATION: Vehicle {track.track_id}")
        
        # Queue violation for broadcasting (thread-safe)
        try:
            manager.violation_queue.put_nowait(violation_data)
        except Exception as e:
            logger.error(f"Failed to queue violation: {e}")
    
    def cleanup_old_tracks(self, current_time, timeout=3.0):
        """Remove inactive tracks"""
        tracks_to_remove = []
        for track_id, track in self.active_tracks.items():
            if len(track.timestamps) > 0 and (current_time - track.timestamps[-1]) > timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
    
    def draw_annotations(self, frame, vehicles):
        """Draw all annotations on frame - only draws detected elements"""
        annotated = frame.copy()
        
        # Draw detected lane lines (only if actually detected)
        if Config.ENABLE_ROAD_LINE_DETECTION and self.lane_detector:
            annotated = self.lane_detector.draw_lanes(annotated)
        
        # Draw detected zebra crossings (only if actually detected)
        if Config.ENABLE_ZEBRA_DETECTION:
            annotated = self.zebra_crossing_detector.draw_zebra_crossing(annotated)
        
        # Draw traffic light info (only if actually detected)
        if Config.ENABLE_TRAFFIC_LIGHT_DETECTION:
            annotated = self.traffic_light_detector.draw_traffic_light_info(annotated)
        
        # Draw vehicle detections
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            track_id = vehicle['track_id']
            class_name = vehicle['class']
            speed = vehicle['speed']
            lane = vehicle['lane']
            violations = vehicle['violations']
            
            # Color based on violations
            if violations:
                color = (0, 0, 255)  # Red for violations
                thickness = 3
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare labels
            label = f"ID:{track_id} {class_name} L{lane+1}"
            speed_label = f"{speed:.1f} km/h"
            
            # Draw labels
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            (w2, h2), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            bg_height = h1 + h2 + 10
            bg_width = max(w1, w2) + 10
            
            cv2.rectangle(annotated, (x1, y1 - bg_height), (x1 + bg_width, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - h2 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(annotated, speed_label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw violation badges
            if violations:
                badge_y = y2 + 20
                for violation in violations:
                    violation_text = violation.upper()
                    (vw, vh), _ = cv2.getTextSize(violation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated, (x1, badge_y), (x1 + vw + 10, badge_y + vh + 5), (0, 0, 255), -1)
                    cv2.putText(annotated, violation_text, (x1 + 5, badge_y + vh),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    badge_y += vh + 10
        
        # Draw statistics panel
        stats_y = 30
        stats_bg_height = 170
        cv2.rectangle(annotated, (5, 5), (420, stats_bg_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (5, 5), (420, stats_bg_height), (255, 255, 255), 2)
        
        # Show signal status only if detected
        signal_status = self.current_signal_state if self.traffic_light_detector.is_detected else "Not Detected"
        
        # Detection status
        detections = []
        if self.traffic_light_detector.is_detected:
            detections.append("Signal")
        if self.lane_detector and self.lane_detector.is_detected:
            detections.append("Lanes")
        if self.zebra_crossing_detector.is_detected:
            detections.append("Zebra")
        detection_str = ", ".join(detections) if detections else "None"
        
        stats = [
            f"Stream {self.stream_id} | FPS: {self.fps:.1f}",
            f"Signal: {signal_status} | Vehicles: {self.total_vehicles}",
            f"Active: {len(self.active_tracks)} | Avg Speed: {self.average_speed:.1f} km/h",
            f"Detected: {detection_str}",
            f"Violations: Speed={self.violation_counts['speed']} RedLight={self.violation_counts['red_light']}",
            f"           StopLine={self.violation_counts['stop_line']} Lane={self.violation_counts['lane_change']}",
            f"           WrongLane={self.violation_counts['wrong_lane']} Distance={self.violation_counts['unsafe_distance']}"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(annotated, stat, (10, stats_y + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        return annotated

# ==================== MULTI-STREAM MANAGER ====================

class MultiStreamManager:
    """Manage multiple video streams"""
    
    def __init__(self):
        self.streams: Dict[int, SingleStreamMonitor] = {}
        self.stream_threads: Dict[int, threading.Thread] = {}
        self.websocket_clients = set()
        self.violation_queue = queue.Queue()  # Thread-safe queue for violations
        logger.info("Multi-Stream Manager initialized")
    
    async def broadcast_violation(self, violation_data: dict):
        """Broadcast violation to all connected WebSocket clients"""
        message = {
            "type": "violation",
            "data": violation_data,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send violation to client: {e}")
                disconnected.add(client)
        
        self.websocket_clients -= disconnected
    
    def start_stream(self, stream_id: int, stream_url: str, save_output: bool = True) -> bool:
        """Start a specific stream"""
        if stream_id in self.streams and self.streams[stream_id].processing:
            logger.warning(f"Stream {stream_id} already running")
            return False
        
        if stream_id >= Config.MAX_STREAMS:
            logger.error(f"Stream ID {stream_id} exceeds maximum {Config.MAX_STREAMS}")
            return False
        
        # Handle YouTube URLs
        if 'youtube.com' in stream_url or 'youtu.be' in stream_url:
            actual_url = get_youtube_stream_url(stream_url)
            if not actual_url:
                return False
            stream_url = actual_url
        
        # Handle webcam
        if stream_url.isdigit():
            stream_url = int(stream_url)
        
        if stream_id not in self.streams:
            self.streams[stream_id] = SingleStreamMonitor(stream_id)
        
        self.streams[stream_id].stream_url = stream_url
        self.streams[stream_id].processing = True
        
        # Setup video writer for output
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Config.OUTPUT_DIR / f"stream{stream_id}_output_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
            self.streams[stream_id].video_writer = cv2.VideoWriter(
                str(output_file), fourcc, Config.OUTPUT_FPS, 
                (Config.INPUT_WIDTH, Config.INPUT_HEIGHT)
            )
            self.streams[stream_id].output_path = output_file
            logger.info(f"Output video will be saved to: {output_file}")
        
        thread = threading.Thread(target=self._process_stream, args=(stream_id, stream_url), daemon=True)
        self.stream_threads[stream_id] = thread
        thread.start()
        
        logger.info(f"Stream {stream_id} started")
        return True
    
    def stop_stream(self, stream_id: int):
        """Stop a specific stream"""
        if stream_id in self.streams:
            self.streams[stream_id].processing = False
            
            # Release video writer
            if self.streams[stream_id].video_writer is not None:
                self.streams[stream_id].video_writer.release()
                logger.info(f"Video saved to: {self.streams[stream_id].output_path}")
            
            logger.info(f"Stream {stream_id} stopped")
    
    def _process_stream(self, stream_id: int, stream_url: str):
        """Process video stream"""
        monitor = self.streams[stream_id]
        cap = cv2.VideoCapture(stream_url)
        
        # Set buffer size for better performance
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Set FPS if it's a file
        if isinstance(stream_url, str) and os.path.isfile(stream_url):
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps > 0:
                frame_delay = 1.0 / original_fps
            else:
                frame_delay = 1.0 / Config.OUTPUT_FPS
        else:
            frame_delay = 1.0 / Config.OUTPUT_FPS
        
        frame_count = 0
        start_time = time.time()
        
        if not cap.isOpened():
            logger.error(f"Stream {stream_id}: Failed to open {stream_url}")
            monitor.processing = False
            return
        
        logger.info(f"Stream {stream_id}: Processing started")
        
        while monitor.processing:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Stream {stream_id}: End of video or failed to read frame")
                break
            
            frame_count += 1
            
            # Process every frame (no skipping for smooth output)
            try:
                monitor.frame_count = frame_count
                monitor.process_frame(frame)
            except Exception as e:
                logger.error(f"Stream {stream_id}: Processing error - {e}")
                import traceback
                traceback.print_exc()
            
            # Calculate FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                monitor.fps = frame_count / elapsed
            
            # Control playback speed for smooth output
            time.sleep(max(0.001, frame_delay - 0.005))  # Slight adjustment for processing time
        
        cap.release()
        if monitor.video_writer is not None:
            monitor.video_writer.release()
        
        logger.info(f"Stream {stream_id} processing stopped. Total frames: {frame_count}")
        logger.info(f"Total violations detected: {sum(monitor.violation_counts.values())}")
    
    def get_all_stats(self):
        """Get statistics from all streams"""
        total_violations = []
        total_vehicles = 0
        active_streams = 0
        
        stream_stats = []
        violation_summary = defaultdict(int)
        
        for stream_id, monitor in self.streams.items():
            if monitor.processing:
                active_streams += 1
            
            total_vehicles += monitor.total_vehicles
            total_violations.extend(monitor.violations)
            
            # Sum violation counts
            for v_type, count in monitor.violation_counts.items():
                violation_summary[v_type] += count
            
            stream_stats.append({
                'stream_id': stream_id,
                'processing': monitor.processing,
                'fps': round(monitor.fps, 1),
                'total_vehicles': monitor.total_vehicles,
                'active_vehicles': len(monitor.active_tracks),
                'average_speed': round(monitor.average_speed, 1),
                'signal_state': monitor.current_signal_state,
                'stream_url': str(monitor.stream_url) if monitor.stream_url else None,
                'violation_counts': dict(monitor.violation_counts),
                'output_path': str(monitor.output_path) if monitor.output_path else None
            })
        
        return {
            'total_streams': len(self.streams),
            'active_streams': active_streams,
            'total_vehicles': total_vehicles,
            'total_violations': len(total_violations),
            'violation_summary': dict(violation_summary),
            'streams': stream_stats,
            'violations': sorted(total_violations, key=lambda x: x['timestamp'], reverse=True)
        }

# ==================== INITIALIZE MANAGER ====================

manager = MultiStreamManager()

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "Complete Traffic Monitoring System API v4.0",
        "version": "4.0.0",
        "max_streams": Config.MAX_STREAMS,
        "features": [
            "Multi-stream support",
            "Vehicle detection and tracking",
            "Speed monitoring and violations",
            "Traffic signal detection (Red/Yellow/Green)",
            "Red light violation detection",
            "Stop line violation detection",
            "Lane violation detection",
            "Wrong-lane detection",
            "Unsafe distance detection",
            "Smooth video output"
        ]
    }

@app.get("/api/health")
async def health():
    stats = manager.get_all_stats()
    return {
        "status": "healthy",
        "active_streams": stats['active_streams'],
        "total_streams": stats['total_streams'],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/start-stream/{stream_id}")
async def start_stream(stream_id: int, stream_url: str, save_output: bool = True):
    """Start a specific stream"""
    if stream_id < 0 or stream_id >= Config.MAX_STREAMS:
        raise HTTPException(status_code=400, detail=f"Stream ID must be between 0 and {Config.MAX_STREAMS-1}")
    
    success = manager.start_stream(stream_id, stream_url, save_output)
    if success:
        return {
            "status": "started",
            "stream_id": stream_id,
            "stream_url": stream_url,
            "save_output": save_output
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to start stream")

@app.post("/api/stop-stream/{stream_id}")
async def stop_stream(stream_id: int):
    """Stop a specific stream"""
    manager.stop_stream(stream_id)
    return {
        "status": "stopped",
        "stream_id": stream_id
    }

@app.post("/api/stop-all-streams")
async def stop_all_streams():
    """Stop all streams"""
    for stream_id in list(manager.streams.keys()):
        manager.stop_stream(stream_id)
    return {"status": "all_streams_stopped"}

@app.post("/api/upload-video/{stream_id}")
async def upload_video(stream_id: int, file: UploadFile = File(...), save_output: bool = True):
    """Upload video file for processing"""
    if stream_id < 0 or stream_id >= Config.MAX_STREAMS:
        raise HTTPException(status_code=400, detail=f"Stream ID must be between 0 and {Config.MAX_STREAMS-1}")
    
    # Check if stream is already running
    if stream_id in manager.streams and manager.streams[stream_id].processing:
        raise HTTPException(status_code=400, detail=f"Stream {stream_id} is already running. Stop it first.")
    
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"stream{stream_id}_{timestamp}{file_extension}"
        file_path = Config.UPLOADS_DIR / safe_filename
        
        # Save uploaded file
        logger.info(f"Saving uploaded file for stream {stream_id}: {file_path}")
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"File saved: {file_path} ({file_size_mb:.2f} MB)")
        
        # Start processing the uploaded video
        success = manager.start_stream(stream_id, str(file_path), save_output)
        
        if success:
            return {
                "status": "success",
                "message": "Video uploaded and processing started",
                "stream_id": stream_id,
                "filename": safe_filename,
                "size_mb": round(file_size_mb, 2),
                "file_path": str(file_path)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start stream after upload")
        
    except Exception as e:
        logger.error(f"Upload error for stream {stream_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive statistics from all streams"""
    return manager.get_all_stats()

@app.get("/api/violations")
async def get_violations(limit: int = 100, violation_type: str = None):
    """Get violations from all streams"""
    stats = manager.get_all_stats()
    violations = stats['violations']
    
    # Filter by type if specified
    if violation_type:
        violations = [v for v in violations if v['violation_type'] == violation_type]
    
    return {
        "total": len(violations),
        "violation_summary": stats['violation_summary'],
        "violations": violations[:limit]
    }

@app.get("/stream/{stream_id}")
async def stream_video(stream_id: int):
    """Get MJPEG stream for a specific stream"""
    if stream_id not in manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    monitor = manager.streams[stream_id]
    
    if not monitor.processing:
        raise HTTPException(status_code=404, detail="Stream not active")
    
    def generate():
        empty_frame_count = 0
        max_empty_frames = 100  # ~3 seconds of waiting
        
        while monitor.processing:
            frame_to_send = None
            
            with monitor.buffer_lock:
                if monitor.current_frame is not None:
                    frame_to_send = monitor.current_frame.copy()
                    empty_frame_count = 0
            
            if frame_to_send is not None:
                ret, buffer = cv2.imencode('.jpg', frame_to_send, 
                                          [cv2.IMWRITE_JPEG_QUALITY, Config.VIDEO_QUALITY])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                empty_frame_count += 1
                if empty_frame_count > max_empty_frames:
                    # Send a placeholder frame
                    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Loading stream...", (500, 360),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    empty_frame_count = 0
            
            time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/{stream_id}/frame")
async def get_single_frame(stream_id: int):
    """Get a single frame from a stream (for polling fallback)"""
    if stream_id not in manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    monitor = manager.streams[stream_id]
    
    with monitor.buffer_lock:
        if monitor.current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', monitor.current_frame, 
                                      [cv2.IMWRITE_JPEG_QUALITY, Config.VIDEO_QUALITY])
            if ret:
                return StreamingResponse(
                    iter([buffer.tobytes()]),
                    media_type="image/jpeg"
                )
    
    # Return placeholder if no frame available
    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for video...", (480, 360),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', placeholder)
    return StreamingResponse(
        iter([buffer.tobytes()]),
        media_type="image/jpeg"
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.websocket_clients.add(websocket)
    
    try:
        await websocket.send_json({
            "type": "connected",
            "message": "WebSocket connected",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Send stats every 2 seconds
            stats = manager.get_all_stats()
            await websocket.send_json({
                "type": "stats_update",
                "data": stats,
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(2)
            
    except WebSocketDisconnect:
        manager.websocket_clients.discard(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.websocket_clients.discard(websocket)

@app.get("/dashboard")
async def dashboard():
    """Serve dashboard"""
    dashboard_path = Path("frontend/dashboard.html")
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")

# Serve static files
try:
    if Path("frontend").exists():
        app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("Complete Traffic Monitoring System v4.0 Started")
    logger.info(f"Maximum Streams: {Config.MAX_STREAMS}")
    logger.info(f"Speed Limit: {Config.SPEED_LIMIT_KMH} km/h")
    logger.info(f"Violation Types: Speed, Red Light, Stop Line, Lane, Wrong Lane, Distance")
    logger.info("=" * 70)
    
    # Start background task to process violation queue
    asyncio.create_task(process_violation_queue())

async def process_violation_queue():
    """Process violations from queue and broadcast to WebSocket clients"""
    logger.info("Violation queue processor started")
    while True:
        try:
            # Check queue for new violations (non-blocking)
            try:
                violation_data = manager.violation_queue.get_nowait()
                await manager.broadcast_violation(violation_data)
            except queue.Empty:
                pass  # No violations to process
            
            # Small delay to prevent CPU spinning
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error processing violation queue: {e}")
            await asyncio.sleep(1)

@app.on_event("shutdown")
async def shutdown_event():
    for stream_id in list(manager.streams.keys()):
        manager.stop_stream(stream_id)
    logger.info("Complete Traffic Monitoring System stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
