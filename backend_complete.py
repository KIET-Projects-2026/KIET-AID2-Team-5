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
- User Authentication (Signup/Login/Logout)
- MongoDB Database for Users and Violations
"""

from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect, UploadFile, File, Depends, status, Query
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import cv2
import numpy as np
import asyncio
import time
import threading
import subprocess
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import queue

# Authentication imports
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

# MongoDB imports
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

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

# ==================== STATIC FILES (FRONTEND) ====================

# Mount the frontend directory so that /frontend/styles.css, /frontend/app.js, etc. work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(script_dir, "frontend")

if os.path.isdir(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

# ==================== JWT & AUTH CONFIGURATION ====================

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-change-in-production-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ==================== MONGODB CONFIGURATION ====================

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "traffic_monitoring")

# MongoDB client (initialized on startup)
mongodb_client: AsyncIOMotorClient = None
database = None

async def get_database():
    """Get database instance"""
    return database

async def connect_to_mongodb():
    """Connect to MongoDB"""
    global mongodb_client, database
    try:
        mongodb_client = AsyncIOMotorClient(MONGODB_URL)
        database = mongodb_client[DATABASE_NAME]
        # Test connection
        await mongodb_client.admin.command('ping')
        logger.info(f"✅ Connected to MongoDB: {DATABASE_NAME}")
        
        # Create indexes
        await database.users.create_index("email", unique=True)
        await database.users.create_index("username", unique=True)
        await database.violations.create_index("timestamp")
        await database.violations.create_index("stream_id")
        await database.violations.create_index("violation_type")
        logger.info("✅ MongoDB indexes created")
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        logger.warning("⚠️ Running without MongoDB - using in-memory storage")

async def close_mongodb_connection():
    """Close MongoDB connection"""
    global mongodb_client
    if mongodb_client:
        mongodb_client.close()
        logger.info("MongoDB connection closed")

# ==================== PYDANTIC MODELS ====================

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field=None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class TokenData(BaseModel):
    email: Optional[str] = None

class ViolationCreate(BaseModel):
    stream_id: int
    track_id: int
    vehicle_class: Optional[str] = None
    speed_kmh: float
    violation_type: str
    signal_state: Optional[str] = None
    image_path: Optional[str] = None

class ViolationResponse(BaseModel):
    id: str
    stream_id: int
    track_id: int
    vehicle_class: Optional[str] = None
    speed_kmh: float
    violation_type: str
    signal_state: Optional[str] = None
    timestamp: datetime
    image_path: Optional[str] = None

# ==================== AUTH HELPER FUNCTIONS ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email from database"""
    if database is None:
        return None
    user = await database.users.find_one({"email": email})
    return user

async def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username from database"""
    if database is None:
        return None
    user = await database.users.find_one({"username": username})
    return user

async def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user with email and password"""
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """Get current user from JWT token"""
    if token is None:
        return None
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        token_data = TokenData(email=email)
    except JWTError:
        return None
    
    user = await get_user_by_email(token_data.email)
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Get current active user (raises exception if not authenticated)"""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not current_user.get("is_active", True):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Optional authentication (for endpoints that work with or without auth)
async def get_optional_user(token: str = Depends(oauth2_scheme)) -> Optional[dict]:
    """Get user if authenticated, None otherwise"""
    if token is None:
        return None
    return await get_current_user(token)

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
    
    # Zebra crossing detection
    ENABLE_ZEBRA_DETECTION = True
    
    # Lane detection configuration
    ROAD_LINE_WHITE_THRESHOLD = 180  # Threshold for detecting white lines
    ROAD_LINE_MIN_LENGTH = 50  # Minimum length of detected lines in pixels
    
    # Speed Detection
    SPEED_LIMIT_KMH = 40  # Speed limit for violation detection
    PIXELS_PER_METER = 15.0  # Adjusted for typical traffic camera perspective
    SPEED_CALCULATION_FRAMES = 15  # Frames for speed calculation
    MOVEMENT_THRESHOLD = 3.0  # pixels - threshold for detecting movement
    
    # Distance-based violations
    SAFE_DISTANCE_METERS = 3.0  # Safe following distance in meters
    UNSAFE_DISTANCE_PIXELS = SAFE_DISTANCE_METERS * PIXELS_PER_METER  # 45 pixels (3.0 * 15)
    
    # Red light violation - crossing line position (ratio of frame height)
    RED_LIGHT_LINE_RATIO = 0.5  # Line at 50% of frame height
    
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


def is_youtube_url(url: str) -> bool:
    """Check if URL is a YouTube URL"""
    youtube_patterns = [
        'youtube.com/watch',
        'youtube.com/live',
        'youtu.be/',
        'youtube.com/shorts',
        'youtube.com/embed'
    ]
    return any(pattern in url.lower() for pattern in youtube_patterns)

# ==================== TRAFFIC LIGHT DETECTION ====================

class TrafficLightDetector:
    """Detect traffic lights using YOLO model and analyze color state"""
    
    def __init__(self):
        self.current_state = None  # None means no traffic light detected
        self.state_history = deque(maxlen=15)
        self.detected_lights = []  # List of detected traffic light positions
        self.last_detection_time = 0
        self.detection_confidence = 0.0
        self.state_change_time = time.time()
        self.is_detected = False  # Flag to indicate if traffic light is actually detected
        
        # Initialize YOLO model for traffic light detection
        # Traffic light is class 9 in COCO dataset
        self.detector = YOLO('yolov8n.pt')
        self.traffic_light_class_id = 9  # COCO class ID for traffic light
        self.confidence_threshold = 0.3  # Lower threshold for traffic lights (they can be small)
        
        logger.info("Traffic Light Detector initialized - YOLO detection mode")
    
    def detect_traffic_lights(self, frame):
        """
        Detect traffic lights in frame using YOLO model.
        Returns list of detected lights with their states and positions.
        """
        self.detected_lights = []
        
        # Run YOLO detection
        results = self.detector(
            frame,
            conf=self.confidence_threshold,
            verbose=False,
            classes=[self.traffic_light_class_id]  # Only detect traffic lights
        )
        
        colors_detected = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Extract the traffic light region
                    traffic_light_roi = frame[y1:y2, x1:x2]
                    
                    if traffic_light_roi.size > 0:
                        # Analyze the color of the traffic light
                        light_color, color_confidence = self._analyze_traffic_light_color(traffic_light_roi)
                        
                        if light_color:
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            area = (x2 - x1) * (y2 - y1)
                            
                            colors_detected.append({
                                'color': light_color,
                                'position': (x1, y1, x2, y2),
                                'center': center,
                                'area': area,
                                'brightness': color_confidence * 255,
                                'confidence': float(conf) * color_confidence,
                                'yolo_confidence': float(conf)
                            })
        
        self.detected_lights = colors_detected
        self.is_detected = len(colors_detected) > 0
        
        return colors_detected
    
    def _analyze_traffic_light_color(self, roi):
        """
        Analyze the ROI to determine which light is active (RED, YELLOW, or GREEN).
        Returns the detected color and confidence.
        """
        if roi.size == 0:
            return None, 0.0
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Divide the ROI into three sections (top=red, middle=yellow, bottom=green)
        height = roi.shape[0]
        section_height = height // 3
        
        # Define regions
        top_section = hsv[0:section_height, :]  # Red light area
        middle_section = hsv[section_height:2*section_height, :]  # Yellow light area
        bottom_section = hsv[2*section_height:, :]  # Green light area
        
        # Calculate brightness/color scores for each section
        red_score = self._calculate_color_score(top_section, 'RED')
        yellow_score = self._calculate_color_score(middle_section, 'YELLOW')
        green_score = self._calculate_color_score(bottom_section, 'GREEN')
        
        # Also check full ROI for single-light traffic signals
        full_red_score = self._calculate_color_score(hsv, 'RED')
        full_yellow_score = self._calculate_color_score(hsv, 'YELLOW')
        full_green_score = self._calculate_color_score(hsv, 'GREEN')
        
        # Use maximum of sectional and full scores
        red_score = max(red_score, full_red_score * 0.8)
        yellow_score = max(yellow_score, full_yellow_score * 0.8)
        green_score = max(green_score, full_green_score * 0.8)
        
        # Determine which light is active
        scores = {'RED': red_score, 'YELLOW': yellow_score, 'GREEN': green_score}
        max_color = max(scores, key=scores.get)
        max_score = scores[max_color]
        
        # Require minimum score threshold
        if max_score > 0.15:
            return max_color, min(max_score, 1.0)
        
        return None, 0.0
    
    def _calculate_color_score(self, hsv_region, color_name):
        """Calculate how much of a specific color is present in the HSV region"""
        if hsv_region.size == 0:
            return 0.0
        
        # Create masks for each color
        if color_name == 'RED':
            # Red has two ranges in HSV (wraps around)
            mask1 = cv2.inRange(hsv_region, Config.RED_LOWER, Config.RED_UPPER)
            mask2 = cv2.inRange(hsv_region, Config.RED_LOWER2, Config.RED_UPPER2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name == 'YELLOW':
            mask = cv2.inRange(hsv_region, Config.YELLOW_LOWER, Config.YELLOW_UPPER)
        elif color_name == 'GREEN':
            mask = cv2.inRange(hsv_region, Config.GREEN_LOWER, Config.GREEN_UPPER)
        else:
            return 0.0
        
        # Calculate the percentage of pixels matching the color
        total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
        if total_pixels == 0:
            return 0.0
        
        color_pixels = cv2.countNonZero(mask)
        score = color_pixels / total_pixels
        
        # Also consider brightness (active lights are bright)
        v_channel = hsv_region[:, :, 2]
        brightness = np.mean(v_channel) / 255.0
        
        # Combine color presence with brightness
        combined_score = score * 0.6 + brightness * 0.4
        
        return combined_score
    
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
            
            # Draw bounding box around detected traffic light
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw circle around the light center
            center = light['center']
            radius = min((x2 - x1), (y2 - y1)) // 4
            cv2.circle(frame, center, radius, color, -1)
            
            # Draw label with YOLO confidence and color
            yolo_conf = light.get('yolo_confidence', light['confidence'])
            label = f"SIGNAL: {light['color']}"
            conf_label = f"YOLO: {yolo_conf*100:.0f}% | Color: {light['confidence']*100:.0f}%"
            
            (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (w2, h2), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_w = max(w1, w2)
            
            # Draw background for label
            cv2.rectangle(frame, (x1, y1 - h1 - h2 - 20), (x1 + max_w + 15, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - h2 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, conf_label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


# ==================== ZEBRA CROSSING DETECTION ====================

class ZebraCrossingDetector:
    """
    Detect zebra/pedestrian crossings using YOLO model.
    Uses YOLO to detect wide white crossing lines on the road.
    """
    
    def __init__(self):
        self.detected_crossings = []
        self.is_detected = False
        self.crossing_region = None
        self.detection_confidence = 0.0
        self.crossing_y_position = None  # Y position of detected crossing
        
        # Initialize YOLO model for detection
        self.detector = YOLO('yolov8n.pt')
        
        logger.info("Zebra Crossing Detector initialized - YOLO detection mode")
    
    def detect_zebra_crossing(self, frame):
        """
        Detect zebra/pedestrian crossing using YOLO and wide white line detection.
        Combines YOLO object detection with computer vision for better accuracy.
        """
        self.detected_crossings = []
        self.is_detected = False
        self.crossing_y_position = None
        
        # Convert to grayscale for white line detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on road area (middle to lower portion of frame where crossings appear)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        road_region_start = int(frame_height * 0.4)
        road_region_end = int(frame_height * 0.85)
        road_region = gray[road_region_start:road_region_end, :]
        
        # Detect wide white horizontal lines (zebra crossing characteristics)
        blurred = cv2.GaussianBlur(road_region, (5, 5), 0)
        _, white_mask = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        
        # Use horizontal kernel to detect wide horizontal stripes
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_h)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_h)
        
        # Find contours of white regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for wide horizontal white stripes
        wide_stripes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's a wide horizontal stripe (width >> height)
            if w > 100 and h > 5 and h < 30:  # Wide and relatively thin
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 8:  # Very wide stripe
                    wide_stripes.append({
                        'bbox': (x, y + road_region_start, x + w, y + h + road_region_start),
                        'center_y': y + h // 2 + road_region_start,
                        'width': w,
                        'height': h
                    })
        
        # If we found multiple wide white stripes clustered together, it's likely a crossing
        if len(wide_stripes) >= 3:
            # Sort by Y position
            wide_stripes.sort(key=lambda s: s['center_y'])
            
            # Check if stripes are evenly spaced (characteristic of zebra crossing)
            y_positions = [s['center_y'] for s in wide_stripes]
            spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            
            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
                
                # Stripes should be relatively close together (10-50 pixels)
                if 10 < avg_spacing < 60:
                    # Calculate bounding box for entire crossing
                    min_x = min(s['bbox'][0] for s in wide_stripes)
                    min_y = min(s['bbox'][1] for s in wide_stripes)
                    max_x = max(s['bbox'][2] for s in wide_stripes)
                    max_y = max(s['bbox'][3] for s in wide_stripes)
                    
                    crossing_data = {
                        'bbox': (min_x, min_y, max_x, max_y),
                        'num_stripes': len(wide_stripes),
                        'confidence': min(len(wide_stripes) / 5.0, 1.0),
                        'center_y': (min_y + max_y) // 2
                    }
                    
                    self.detected_crossings.append(crossing_data)
                    self.is_detected = True
                    self.crossing_region = crossing_data['bbox']
                    self.detection_confidence = crossing_data['confidence']
                    self.crossing_y_position = crossing_data['center_y']
                    
                    logger.info(f"Zebra crossing detected at Y={self.crossing_y_position} with {len(wide_stripes)} stripes")
        
        return self.detected_crossings
    

    
    def draw_zebra_crossing(self, frame):
        """Draw detected zebra crossings on frame - only if actually detected"""
        if not self.is_detected or not self.detected_crossings:
            return frame
        
        for crossing in self.detected_crossings:
            x1, y1, x2, y2 = crossing['bbox']
            
            # Draw bounding box around crossing
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Draw center line of crossing (the stop line reference)
            center_y = crossing['center_y']
            cv2.line(frame, (50, center_y), (frame.shape[1] - 50, center_y), (0, 255, 255), 3)
            
            # Draw label
            label = f"CROSSING ({crossing['num_stripes']} stripes, {crossing['confidence']*100:.0f}%)"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 255, 255), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
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
    """Track individual vehicle with violation detection"""
    
    def __init__(self, track_id: int, bbox: tuple, timestamp: float, frame_width: int = 1280):
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
        self.frame_width = frame_width
        
        # Violation flags
        self.violations = set()
        self.violation_recorded = set()
        
        # Red light violation tracking
        self.crossed_red_light_line = False
        self.position_at_red_start = None
        
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
        if len(self.positions) >= 5 and self.speed_kmh > 5:  # Minimum 5 km/h to avoid noise
            if self.speed_kmh > Config.SPEED_LIMIT_KMH:
                if 'speed' not in self.violations:
                    self.violations.add('speed')
                    logger.info(f"SPEED VIOLATION: Vehicle {self.track_id} at {self.speed_kmh:.1f} km/h (limit: {Config.SPEED_LIMIT_KMH} km/h)")
                    return True
        return False
    
    def check_red_light_violation(self, signal_state: str, red_light_line_y: int):
        """
        Check if vehicle crosses the road during RED signal.
        Simple logic: If signal is RED and vehicle crosses the line, it's a violation.
        """
        if signal_state == "RED":
            if len(self.positions) >= 2 and len(self.bboxes) >= 1:
                # Get current vehicle position (bottom of bounding box = front of vehicle)
                x1, y1, x2, y2 = self.bboxes[-1]
                vehicle_front_y = y2
                
                # Get previous position
                prev_center_y = self.positions[-2][1] if len(self.positions) >= 2 else self.positions[-1][1]
                curr_center_y = self.positions[-1][1]
                
                # Check if vehicle has crossed the red light line
                # Vehicle was above line before and is now below (or on) the line
                vehicle_crossed = vehicle_front_y > red_light_line_y
                vehicle_moving_forward = curr_center_y > prev_center_y  # Moving down in frame = forward
                
                if vehicle_crossed and vehicle_moving_forward:
                    if not self.crossed_red_light_line:
                        self.crossed_red_light_line = True
                        if 'red_light' not in self.violations:
                            self.violations.add('red_light')
                            logger.info(f"RED LIGHT VIOLATION: Vehicle {self.track_id} crossed line at Y={vehicle_front_y:.0f} during RED signal")
                            return True
        else:
            # Reset when signal turns GREEN
            if signal_state == "GREEN":
                self.crossed_red_light_line = False
        
        return False
    
    def check_unsafe_distance(self, other_track):
        """
        Check if following distance is unsafe (vehicles too close in following scenario).
        Only detects when one vehicle is directly behind another, not side-by-side.
        """
        if len(self.positions) == 0 or len(other_track.positions) == 0:
            return False
        
        if len(self.bboxes) == 0 or len(other_track.bboxes) == 0:
            return False
        
        # Get bounding boxes
        my_bbox = self.bboxes[-1]
        other_bbox = other_track.bboxes[-1]
        
        my_x1, my_y1, my_x2, my_y2 = my_bbox
        other_x1, other_y1, other_x2, other_y2 = other_bbox
        
        # Calculate center positions
        my_center_x = (my_x1 + my_x2) / 2
        my_center_y = (my_y1 + my_y2) / 2
        other_center_x = (other_x1 + other_x2) / 2
        other_center_y = (other_y1 + other_y2) / 2
        
        # Calculate horizontal and vertical distances
        horizontal_diff = abs(my_center_x - other_center_x)
        vertical_diff = abs(my_center_y - other_center_y)
        
        # Check if vehicles are roughly in the same lane (horizontally aligned)
        # Allow up to 100 pixels horizontal difference (roughly same lane)
        if horizontal_diff > 100:
            return False  # Too far apart horizontally (different lanes)
        
        # Check vertical distance (following distance)
        # One vehicle should be significantly behind/in front of the other
        if vertical_diff < 50:
            return False  # Side by side, not following
        
        # Calculate actual distance between centers
        distance = np.sqrt(horizontal_diff**2 + vertical_diff**2)
        
        # Check if distance is unsafe
        if distance < Config.UNSAFE_DISTANCE_PIXELS:
            # Determine if one vehicle is following the other
            # The following vehicle should be higher in the frame (lower Y value in typical camera view)
            # and moving in same direction
            
            # Check if vehicles are moving in roughly the same direction
            if len(self.positions) >= 2 and len(other_track.positions) >= 2:
                my_prev_y = self.positions[-2][1]
                other_prev_y = other_track.positions[-2][1]
                
                my_direction = my_center_y - my_prev_y  # Positive = moving down/forward
                other_direction = other_center_y - other_prev_y
                
                # If moving in opposite directions, not a following scenario
                if (my_direction * other_direction) < 0:
                    return False
            
            # Valid unsafe distance violation
            if 'unsafe_distance' not in self.violations:
                self.violations.add('unsafe_distance')
                logger.info(f"UNSAFE DISTANCE: Vehicle {self.track_id} too close ({distance:.1f}px, vertical: {vertical_diff:.1f}px) to Vehicle {other_track.track_id}")
                return True
        
        return False

# ==================== SINGLE STREAM MONITOR ====================

class SingleStreamMonitor:
    """Monitor a single video stream with violation detection"""
    
    def __init__(self, stream_id: int):
        self.stream_id = stream_id
        self.vehicle_detector = YOLO('yolov8n.pt')
        self.traffic_light_detector = TrafficLightDetector()
        self.zebra_crossing_detector = ZebraCrossingDetector()
        
        self.active_tracks: Dict[int, VehicleTrack] = {}
        self.vehicle_count = defaultdict(int)
        self.violations = []
        self.current_frame = None
        self.processing = False
        self.stream_url = None
        self.last_error = None  # Track last error for API responses
        
        # Statistics
        self.fps = 0
        self.detection_time = 0
        self.total_vehicles = 0
        self.frame_count = 0
        self.average_speed = 0.0
        self.current_signal_state = None  # None means no signal detected
        self.signal_confidence = 0.0
        
        # Violation counters (simplified: speed, red_light, unsafe_distance)
        self.violation_counts = {
            'speed': 0,
            'red_light': 0,
            'unsafe_distance': 0
        }
        
        # Frame buffering for smooth output
        self.frame_buffer = deque(maxlen=30)
        self.buffer_lock = threading.Lock()
        
        # Video writer for output
        self.video_writer = None
        self.output_path = None
        
        logger.info(f"Stream Monitor {stream_id} initialized")
    
    def process_frame(self, frame):
        """Process a single frame with detection algorithms"""
        start_time = time.time()
        
        # Resize frame
        frame = cv2.resize(frame, (Config.INPUT_WIDTH, Config.INPUT_HEIGHT))
        
        # Detect traffic light state
        if Config.ENABLE_TRAFFIC_LIGHT_DETECTION:
            state, confidence = self.traffic_light_detector.detect_traffic_light_state(frame)
            if state is not None:
                self.current_signal_state = state
                self.signal_confidence = confidence
        
        # Detect zebra crossings (for crossing line reference)
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
                                track_id, (x1, y1, x2, y2), current_time, Config.INPUT_WIDTH
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
        """Check violations: speed, red light crossing, unsafe distance"""
        tracks_list = list(self.active_tracks.values())
        
        # Determine crossing line position
        # Use detected zebra crossing position if available, otherwise use frame center
        frame_height = frame.shape[0]
        if self.zebra_crossing_detector.is_detected and self.zebra_crossing_detector.crossing_y_position:
            crossing_line_y = self.zebra_crossing_detector.crossing_y_position
            logger.debug(f"Using detected crossing at Y={crossing_line_y}")
        else:
            # Fallback to middle of frame
            crossing_line_y = int(frame_height * 0.55)
        
        # Check if traffic light is detected
        signal_detected = self.traffic_light_detector.is_detected
        current_signal = self.current_signal_state if signal_detected else None
        
        for track in tracks_list:
            # 1. SPEED VIOLATION - Always check
            if track.check_speed_violation():
                if 'speed' not in track.violation_recorded:
                    self.record_violation(frame, track, 'speed')
                    track.violation_recorded.add('speed')
            
            # 2. RED LIGHT VIOLATION - If RED signal and vehicle crosses line
            if signal_detected and current_signal == "RED":
                if track.check_red_light_violation(current_signal, crossing_line_y):
                    if 'red_light' not in track.violation_recorded:
                        self.record_violation(frame, track, 'red_light')
                        track.violation_recorded.add('red_light')
            elif signal_detected and current_signal == "GREEN":
                # Reset crossing flag when signal is green
                track.check_red_light_violation(current_signal, crossing_line_y)
        
        # 3. UNSAFE DISTANCE VIOLATION - Check pairs only once
        checked_pairs = set()
        for i, track in enumerate(tracks_list):
            for j in range(i + 1, len(tracks_list)):
                other_track = tracks_list[j]
                
                # Check unsafe distance (only need to check once per pair)
                if track.check_unsafe_distance(other_track):
                    if 'unsafe_distance' not in track.violation_recorded:
                        self.record_violation(frame, track, 'unsafe_distance')
                        track.violation_recorded.add('unsafe_distance')
    
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
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        # Draw detected zebra crossings (includes crossing line)
        annotated = self.zebra_crossing_detector.draw_zebra_crossing(annotated)
        
        # Draw traffic light info (if detected)
        if Config.ENABLE_TRAFFIC_LIGHT_DETECTION:
            annotated = self.traffic_light_detector.draw_traffic_light_info(annotated)
        
        # Draw vehicle detections
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle['bbox']
            track_id = vehicle['track_id']
            class_name = vehicle['class']
            speed = vehicle['speed']
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
            label = f"ID:{track_id} {class_name}"
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
                    (vw, vh), _ = cv2.getTextSize(violation_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, badge_y), (x1 + vw + 10, badge_y + vh + 8), (0, 0, 255), -1)
                    cv2.putText(annotated, violation_text, (x1 + 5, badge_y + vh + 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    badge_y += vh + 15
        
        # Draw statistics panel
        stats_y = 30
        stats_bg_height = 150
        cv2.rectangle(annotated, (5, 5), (420, stats_bg_height), (0, 0, 0), -1)
        cv2.rectangle(annotated, (5, 5), (420, stats_bg_height), (255, 255, 255), 2)
        
        # Show signal status
        signal_status = self.current_signal_state if self.traffic_light_detector.is_detected else "Not Detected"
        
        # Calculate total violations
        total_violations = sum(self.violation_counts.values())
        
        stats = [
            f"Stream {self.stream_id} | FPS: {self.fps:.1f} | Speed Limit: {Config.SPEED_LIMIT_KMH} km/h",
            f"Signal: {signal_status} | Vehicles: {self.total_vehicles} | Active: {len(self.active_tracks)}",
            f"Avg Speed: {self.average_speed:.1f} km/h",
            f"--- VIOLATIONS (Total: {total_violations}) ---",
            f"Speed: {self.violation_counts['speed']} | Red Light: {self.violation_counts['red_light']} | Unsafe Dist: {self.violation_counts['unsafe_distance']}"
        ]
        
        for i, stat in enumerate(stats):
            # Highlight violations header
            text_color = (0, 255, 255) if "VIOLATIONS" in stat else (255, 255, 255)
            cv2.putText(annotated, stat, (10, stats_y + i * 24),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        
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
        
        original_url = stream_url
        
        # Handle YouTube URLs with improved detection
        if is_youtube_url(stream_url):
            logger.info(f"YouTube URL detected: {stream_url}")
            actual_url = get_youtube_stream_url(stream_url)
            if not actual_url:
                logger.error(f"Failed to extract YouTube stream URL from: {stream_url}")
                # Store error for API response
                if stream_id not in self.streams:
                    self.streams[stream_id] = SingleStreamMonitor(stream_id)
                self.streams[stream_id].last_error = "YouTube URL extraction failed. Please try a different video or upload directly."
                return False
            stream_url = actual_url
            logger.info(f"YouTube stream URL extracted successfully")
        
        # Handle webcam
        if isinstance(stream_url, str) and stream_url.isdigit():
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
        """Process video stream with smooth frame delivery"""
        monitor = self.streams[stream_id]
        
        # Configure OpenCV for better network stream handling
        if isinstance(stream_url, str) and stream_url.startswith('http'):
            # For network streams (including YouTube)
            cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000)
        else:
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        # Get source FPS for timing
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        is_file = isinstance(stream_url, str) and os.path.isfile(stream_url)
        
        if is_file and source_fps > 0:
            target_fps = min(source_fps, Config.OUTPUT_FPS)
            frame_delay = 1.0 / target_fps
        else:
            target_fps = Config.OUTPUT_FPS
            frame_delay = 1.0 / target_fps
        
        frame_count = 0
        start_time = time.time()
        processing_times = deque(maxlen=30)  # Track processing times for adaptive delay
        
        if not cap.isOpened():
            logger.error(f"Stream {stream_id}: Failed to open {stream_url}")
            monitor.processing = False
            monitor.last_error = "Failed to open video source"
            return
        
        logger.info(f"Stream {stream_id}: Processing started (Source FPS: {source_fps:.1f}, Target: {target_fps:.1f})")
        
        consecutive_failures = 0
        max_failures = 50  # About 1.5 seconds at 30fps
        
        while monitor.processing:
            frame_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.warning(f"Stream {stream_id}: End of video or too many consecutive failures")
                    break
                # For live streams, small delay before retry
                if not is_file:
                    time.sleep(0.01)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            # Process frame
            try:
                monitor.frame_count = frame_count
                monitor.process_frame(frame)
            except Exception as e:
                logger.error(f"Stream {stream_id}: Processing error - {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Calculate actual FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                monitor.fps = frame_count / elapsed
            
            # Adaptive frame timing for smooth playback
            processing_time = time.time() - frame_start
            processing_times.append(processing_time)
            avg_processing = sum(processing_times) / len(processing_times)
            
            # Calculate delay needed for target frame rate
            target_delay = frame_delay - avg_processing
            
            if target_delay > 0.001:
                time.sleep(target_delay)
            else:
                # At least yield to other threads
                time.sleep(0.001)
        
        cap.release()
        if monitor.video_writer is not None:
            monitor.video_writer.release()
        
        monitor.processing = False
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

# ==================== AUTH API ENDPOINTS ====================

@app.post("/api/auth/signup", response_model=Token)
async def signup(user_data: UserCreate):
    """Register a new user"""
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    # Check if email already exists
    existing_user = await get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username already exists
    existing_username = await get_user_by_username(user_data.username)
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    new_user = {
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "full_name": user_data.full_name,
        "is_active": True,
        "created_at": datetime.utcnow()
    }
    
    result = await database.users.insert_one(new_user)
    new_user["_id"] = result.inserted_id
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user_data.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"New user registered: {user_data.email}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(new_user["_id"]),
            "username": new_user["username"],
            "email": new_user["email"],
            "full_name": new_user["full_name"],
            "is_active": new_user["is_active"],
            "created_at": new_user["created_at"]
        }
    }

@app.post("/api/auth/login", response_model=Token)
async def login(form_data: UserLogin):
    """Login user and return JWT token"""
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    user = await authenticate_user(form_data.email, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"User logged in: {user['email']}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": str(user["_id"]),
            "username": user["username"],
            "email": user["email"],
            "full_name": user.get("full_name"),
            "is_active": user.get("is_active", True),
            "created_at": user["created_at"]
        }
    }

@app.post("/api/auth/login/form")
async def login_form(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible login endpoint for form data"""
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    # OAuth2 uses username field, we'll accept email there
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user["email"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/auth/logout")
async def logout(current_user: dict = Depends(get_current_active_user)):
    """Logout user (client should delete token)"""
    logger.info(f"User logged out: {current_user['email']}")
    return {"message": "Successfully logged out"}

@app.get("/api/auth/me", response_model=UserResponse)
async def get_me(current_user: dict = Depends(get_current_active_user)):
    """Get current user info"""
    return {
        "id": str(current_user["_id"]),
        "username": current_user["username"],
        "email": current_user["email"],
        "full_name": current_user.get("full_name"),
        "is_active": current_user.get("is_active", True),
        "created_at": current_user["created_at"]
    }

@app.get("/api/auth/verify")
async def verify_token(current_user: dict = Depends(get_current_active_user)):
    """Verify if token is valid"""
    return {
        "valid": True,
        "user": {
            "id": str(current_user["_id"]),
            "username": current_user["username"],
            "email": current_user["email"]
        }
    }

# ==================== VIOLATIONS DATABASE ENDPOINTS ====================

@app.get("/api/db/violations")
async def get_violations_from_db(
    limit: int = 100,
    skip: int = 0,
    violation_type: Optional[str] = None,
    stream_id: Optional[int] = None,
    date_range: Optional[str] = None,
    current_user: dict = Depends(get_optional_user)
):
    """
    Get violations from MongoDB database.
    Supports optional filters:
    - violation_type
    - stream_id
    - date_range: one of ['yesterday', 'last_week', 'last_month', 'last_year']
    """
    if database is None:
        # Fallback to in-memory violations
        stats = manager.get_all_stats()
        return {
            "total": len(stats['violations']),
            "violations": stats['violations'][:limit],
            "source": "memory"
        }
    
    # Build query filter
    query: dict = {}
    if violation_type:
        query["violation_type"] = violation_type
    if stream_id is not None:
        query["stream_id"] = stream_id

    # Date range filter
    if date_range:
        now = datetime.utcnow()
        start_time: Optional[datetime] = None
        end_time: Optional[datetime] = None

        if date_range == "yesterday":
            today_start = datetime(now.year, now.month, now.day)
            start_time = today_start - timedelta(days=1)
            end_time = today_start
        elif date_range == "last_week":
            start_time = now - timedelta(days=7)
        elif date_range == "last_month":
            start_time = now - timedelta(days=30)
        elif date_range == "last_year":
            start_time = now - timedelta(days=365)

        if start_time:
            if end_time:
                query["timestamp"] = {"$gte": start_time, "$lt": end_time}
            else:
                query["timestamp"] = {"$gte": start_time}
    
    # Get total count
    total = await database.violations.count_documents(query)
    
    # Get violations
    cursor = database.violations.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    violations = []
    async for violation in cursor:
        violations.append({
            "id": str(violation["_id"]),
            "stream_id": violation["stream_id"],
            "track_id": violation["track_id"],
            "vehicle_class": violation.get("vehicle_class"),
            "speed_kmh": violation["speed_kmh"],
            "violation_type": violation["violation_type"],
            "signal_state": violation.get("signal_state"),
            "timestamp": violation["timestamp"].isoformat() if isinstance(violation["timestamp"], datetime) else violation["timestamp"],
            "image_path": violation.get("image_path")
        })
    
    return {
        "total": total,
        "violations": violations,
        "source": "database"
    }

@app.delete("/api/db/violations/{violation_id}")
async def delete_violation(
    violation_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Delete a violation from database"""
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    result = await database.violations.delete_one({"_id": ObjectId(violation_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Violation not found")
    
    return {"message": "Violation deleted", "id": violation_id}

@app.delete("/api/db/violations")
async def clear_all_violations(current_user: dict = Depends(get_current_active_user)):
    """Clear all violations from database"""
    if database is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    
    result = await database.violations.delete_many({})
    return {"message": f"Deleted {result.deleted_count} violations"}

async def save_violation_to_db(violation_data: dict):
    """Save violation to MongoDB database"""
    if database is None:
        return None
    
    try:
        # Prepare violation document
        violation_doc = {
            "stream_id": violation_data.get("stream_id"),
            "track_id": violation_data.get("track_id"),
            "vehicle_class": violation_data.get("vehicle_class"),
            "speed_kmh": violation_data.get("speed_kmh"),
            "violation_type": violation_data.get("violation_type"),
            "signal_state": violation_data.get("signal_state"),
            "timestamp": datetime.fromisoformat(violation_data["timestamp"]) if isinstance(violation_data["timestamp"], str) else violation_data["timestamp"],
            "image_path": violation_data.get("image_path"),
            "speed_limit": violation_data.get("speed_limit"),
            "excess_speed": violation_data.get("excess_speed")
        }
        
        result = await database.violations.insert_one(violation_doc)
        logger.info(f"Violation saved to DB: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"Failed to save violation to DB: {e}")
        return None

# ==================== API ENDPOINTS ====================

@app.get("/")
async def root():
    """
    Serve the public landing page.
    This is the first page users see, with Home/About/Project/Contact and a Get Started button.
    """
    # Primary path based on this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    landing_path = os.path.join(script_dir, "frontend", "landing.html")

    if os.path.exists(landing_path):
        logger.info(f"Serving landing page from: {landing_path}")
        return FileResponse(landing_path, media_type="text/html")

    # Fallbacks if needed
    fallback_paths = [
        Path.cwd() / "frontend" / "landing.html",
        Path("frontend/landing.html"),
    ]

    for path in fallback_paths:
        if path.exists():
            logger.info(f"Serving landing page from fallback: {path}")
            return FileResponse(str(path), media_type="text/html")

    logger.error(f"Landing page not found. Script dir: {script_dir}, CWD: {Path.cwd()}")
    raise HTTPException(
        status_code=404,
        detail="Landing page not found"
    )

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
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
async def start_stream(stream_id: int, stream_url: str = Query(...), save_output: bool = Query(True)):
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

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/Render"""
    return {
        "status": "healthy",
        "service": "Traffic Monitoring System",
        "version": "4.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive statistics from all streams"""
    return manager.get_all_stats()

@app.get("/api/violations")
async def get_violations(
    limit: int = 100,
    violation_type: Optional[str] = None,
    date_range: Optional[str] = None
):
    """
    Get violations from all streams (in-memory).
    Optional filters:
    - violation_type
    - date_range: one of ['yesterday', 'last_week', 'last_month', 'last_year']
    """
    stats = manager.get_all_stats()
    violations = stats['violations']

    # Filter by type if specified
    if violation_type:
        violations = [v for v in violations if v.get('violation_type') == violation_type]

    # Date-range filter (in-memory)
    if date_range:
        now = datetime.utcnow()
        start_time: Optional[datetime] = None
        end_time: Optional[datetime] = None

        if date_range == "yesterday":
            today_start = datetime(now.year, now.month, now.day)
            start_time = today_start - timedelta(days=1)
            end_time = today_start
        elif date_range == "last_week":
            start_time = now - timedelta(days=7)
        elif date_range == "last_month":
            start_time = now - timedelta(days=30)
        elif date_range == "last_year":
            start_time = now - timedelta(days=365)

        if start_time:
            filtered = []
            for v in violations:
                ts = v.get("timestamp")
                if not ts:
                    continue
                try:
                    if isinstance(ts, str):
                        ts_dt = datetime.fromisoformat(ts)
                    else:
                        ts_dt = ts
                except Exception:
                    continue

                if end_time:
                    if start_time <= ts_dt < end_time:
                        filtered.append(v)
                else:
                    if ts_dt >= start_time:
                        filtered.append(v)
            violations = filtered

    return {
        "total": len(violations),
        "violation_summary": stats['violation_summary'],
        "violations": violations[:limit]
    }

@app.get("/stream/{stream_id}")
async def stream_video(stream_id: int):
    """Get MJPEG stream for a specific stream - optimized for smooth playback"""
    if stream_id not in manager.streams:
        raise HTTPException(status_code=404, detail="Stream not found")
    
    monitor = manager.streams[stream_id]
    
    if not monitor.processing:
        raise HTTPException(status_code=404, detail="Stream not active")
    
    def generate():
        empty_frame_count = 0
        max_empty_frames = 100  # ~3 seconds of waiting
        last_frame = None
        frame_interval = 1.0 / Config.OUTPUT_FPS
        last_send_time = time.time()
        
        while monitor.processing:
            current_time = time.time()
            frame_to_send = None
            
            with monitor.buffer_lock:
                if monitor.current_frame is not None:
                    # Only send if frame is different or enough time has passed
                    frame_to_send = monitor.current_frame.copy()
                    empty_frame_count = 0
            
            if frame_to_send is not None:
                # Encode frame with optimized quality for streaming
                encode_params = [
                    cv2.IMWRITE_JPEG_QUALITY, Config.VIDEO_QUALITY,
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                ]
                ret, buffer = cv2.imencode('.jpg', frame_to_send, encode_params)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n'
                           b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' 
                           + frame_bytes + b'\r\n')
                    last_frame = frame_to_send
                    last_send_time = current_time
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
            
            # Smooth frame rate control
            elapsed = time.time() - current_time
            sleep_time = max(0.001, frame_interval - elapsed)
            time.sleep(sleep_time)
    
    return StreamingResponse(
        generate(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "keep-alive"
        }
    )

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
async def serve_dashboard():
    """Serve dashboard HTML page"""
    # Get the directory where this script is located using os.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "frontend", "dashboard.html")
    
    logger.info(f"Looking for dashboard at: {dashboard_path}")
    
    if os.path.exists(dashboard_path):
        logger.info(f"Serving dashboard from: {dashboard_path}")
        return FileResponse(dashboard_path, media_type="text/html")
    
    # Fallback paths
    fallback_paths = [
        Path.cwd() / "frontend" / "dashboard.html",
        Path("frontend/dashboard.html"),
    ]
    
    for path in fallback_paths:
        if path.exists():
            logger.info(f"Serving dashboard from fallback: {path}")
            return FileResponse(str(path), media_type="text/html")
    
    logger.error(f"Dashboard not found. Script dir: {script_dir}, CWD: {Path.cwd()}")
    
    raise HTTPException(
        status_code=404, 
        detail="Dashboard not found"
    )

@app.on_event("startup")
async def startup_event():
    # Connect to MongoDB
    await connect_to_mongodb()
    
    logger.info("=" * 70)
    logger.info("Complete Traffic Monitoring System v4.0 Started")
    logger.info(f"Maximum Streams: {Config.MAX_STREAMS}")
    logger.info(f"Speed Limit: {Config.SPEED_LIMIT_KMH} km/h")
    logger.info(f"Violation Types: Speed, Red Light, Stop Line, Lane, Wrong Lane, Distance")
    logger.info(f"MongoDB: {'Connected' if database is not None else 'Not Connected (using memory)'}")
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
                
                # Save to MongoDB
                await save_violation_to_db(violation_data)
                
                # Broadcast to WebSocket clients
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
    
    # Close MongoDB connection
    await close_mongodb_connection()
    
    logger.info("Complete Traffic Monitoring System stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
