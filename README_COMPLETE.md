# 🚦 Complete Traffic Monitoring System v4.0

## Advanced AI-Powered Traffic Violation Detection System

A comprehensive real-time traffic monitoring solution with **complete violation detection** including vehicle tracking, speed monitoring, traffic signal detection, and multiple violation types using YOLOv8.

---

## ✨ Complete Feature Set

### 🎯 Vehicle Detection & Tracking
- **Real-time Detection** - Cars, trucks, buses, motorcycles, bicycles
- **Persistent Tracking** - ByteTrack algorithm for stable vehicle tracking
- **Speed Calculation** - Real-time speed estimation in km/h
- **Lane Assignment** - Automatic lane detection and vehicle positioning

### 🚨 Comprehensive Violation Detection

#### 1. **Speed Violations**
- Automatic detection of vehicles exceeding speed limit
- Configurable speed limit (default: 60 km/h)
- Records excess speed amount

#### 2. **Red Light Violations**
- Automatic traffic signal detection (Red/Yellow/Green)
- Detects vehicles moving during red light
- Tracks signal state when violation occurs

#### 3. **Stop Line Violations**
- Detects vehicles crossing stop line during red/yellow signal
- Configurable stop line position
- Visual stop line marking on video

#### 4. **Lane Violations**
- Detects frequent lane changes
- Monitors lane discipline
- Multi-lane support (configurable)

#### 5. **Wrong Lane Detection**
- Identifies vehicles in wrong lanes
- Configurable lane direction rules
- Visual lane markers

#### 6. **Unsafe Distance Violations**
- Calculates distance between vehicles
- Detects unsafe following distance
- Configurable safe distance threshold

### 📹 Video Processing
- **Smooth Video Output** - Fixed frame rate output (30 FPS)
- **High Quality** - Configurable video quality
- **Real-time Streaming** - MJPEG streaming for web dashboard
- **Video Recording** - Automatic saving of processed videos
- **Violation Screenshots** - Automatic capture of violation moments

### 🌐 Web Dashboard
- Real-time video streaming
- Live statistics and analytics
- Violation notifications
- WebSocket-based updates
- Multi-stream support (up to 4 streams)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam, video file, or YouTube URL
- 4GB+ RAM recommended
- GPU optional (CPU works fine)

### Installation

1. **Navigate to project directory**
```powershell
cd p:\projects\traffic-monitoring-system
```

2. **Create and activate virtual environment** (recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Verify YOLOv8 model**
The `yolov8n.pt` model should be in the project directory. If not, it will be downloaded automatically on first run.

---

## 🎮 Running the System

### Method 1: Simple Run Script (Recommended)
```powershell
python run_complete_system.py
```

### Method 2: Direct Execution
```powershell
python backend_complete.py
```

### Method 3: Using uvicorn
```powershell
uvicorn backend_complete:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at:
- **Dashboard**: http://localhost:8000/dashboard
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

---

## 📝 Usage Examples

### Using the Web Dashboard

1. **Open the dashboard**
   ```
   http://localhost:8000/dashboard
   ```

2. **Upload a video**
   - Click on "Upload Video" button
   - Select your traffic video file
   - Processing starts automatically

3. **Monitor violations**
   - View real-time statistics
   - Check violation list
   - Review violation images

### Using the API

#### Start Processing a Video File
```python
import requests

# Upload and start processing
files = {'file': open('traffic_video.mp4', 'rb')}
response = requests.post(
    'http://localhost:8000/api/upload-video/0',
    files=files,
    params={'save_output': True}
)
print(response.json())
```

#### Start Processing a Webcam
```python
import requests

response = requests.post(
    'http://localhost:8000/api/start-stream/0',
    params={'stream_url': '0'}  # 0 for default webcam
)
print(response.json())
```

#### Start Processing a YouTube Video
```python
import requests

response = requests.post(
    'http://localhost:8000/api/start-stream/0',
    params={'stream_url': 'https://www.youtube.com/watch?v=VIDEO_ID'}
)
print(response.json())
```

#### Get Statistics
```python
import requests

response = requests.get('http://localhost:8000/api/stats')
stats = response.json()
print(f"Total Vehicles: {stats['total_vehicles']}")
print(f"Total Violations: {stats['total_violations']}")
print(f"Violation Summary: {stats['violation_summary']}")
```

#### Get Violations
```python
import requests

# Get all violations
response = requests.get('http://localhost:8000/api/violations')
violations = response.json()

# Get specific violation type
response = requests.get(
    'http://localhost:8000/api/violations',
    params={'violation_type': 'speed', 'limit': 50}
)
speed_violations = response.json()
```

#### Stop Stream
```python
import requests

response = requests.post('http://localhost:8000/api/stop-stream/0')
print(response.json())
```

---

## ⚙️ Configuration

Edit the `Config` class in `backend_complete.py` to customize:

```python
class Config:
    # Detection Settings
    CONFIDENCE_THRESHOLD = 0.45
    INPUT_WIDTH = 1280
    INPUT_HEIGHT = 720
    
    # Speed Settings
    SPEED_LIMIT_KMH = 60
    PIXELS_PER_METER = 8.0
    
    # Distance Settings
    SAFE_DISTANCE_METERS = 3.0
    
    # Lane Settings
    ENABLE_LANE_DETECTION = True
    NUM_LANES = 3
    
    # Stop Line Settings
    STOP_LINE_Y = 400  # Y coordinate
    
    # Video Output Settings
    OUTPUT_FPS = 30
    VIDEO_QUALITY = 90
    
    # Traffic Light Detection
    ENABLE_TRAFFIC_LIGHT_DETECTION = True
```

---

## 📊 Output Files

All output files are saved in the `data/` directory:

```
data/
├── uploads/          # Uploaded video files
├── output/           # Processed video files with annotations
├── violations/       # Violation screenshots
└── logs/            # System logs
```

### Processed Videos
- Location: `data/output/stream{id}_output_{timestamp}.mp4`
- Contains all annotations and violation markers
- Fixed 30 FPS for smooth playback
- Can be played in any video player

### Violation Images
- Location: `data/violations/stream{id}_{type}_{id}_{timestamp}.jpg`
- High-quality screenshots at violation moment
- Includes all annotations

---

## 🎯 Violation Types Explained

### 1. Speed Violation (`speed`)
**Triggered when**: Vehicle speed exceeds configured limit
**Data captured**: 
- Vehicle speed
- Speed limit
- Excess speed amount

### 2. Red Light Violation (`red_light`)
**Triggered when**: Vehicle moves significantly during red signal
**Data captured**:
- Signal state (RED)
- Vehicle movement distance
- Signal crossing time

### 3. Stop Line Violation (`stop_line`)
**Triggered when**: Vehicle crosses stop line during red/yellow signal
**Data captured**:
- Signal state when crossed
- Stop line position
- Vehicle position

### 4. Lane Violation (`lane_change`)
**Triggered when**: Vehicle makes frequent lane changes
**Data captured**:
- Number of lane changes
- Lane transition history

### 5. Wrong Lane Violation (`wrong_lane`)
**Triggered when**: Vehicle is in incorrect lane for traffic direction
**Data captured**:
- Current lane
- Expected lane direction

### 6. Unsafe Distance Violation (`unsafe_distance`)
**Triggered when**: Following distance is less than safe threshold
**Data captured**:
- Distance between vehicles
- Safe distance threshold
- Both vehicle IDs

---

## 🔧 Troubleshooting

### Video Not Playing Smoothly
✅ **FIXED in v4.0** - The system now:
- Uses fixed frame rate (30 FPS)
- Properly controls frame timing
- Uses optimized video codec
- No frame skipping during output

### Traffic Light Not Detected
- System auto-detects traffic lights in upper 30% of frame
- Ensure traffic light is visible in video
- Check lighting conditions
- Adjust detection thresholds in Config if needed

### Low FPS
- Reduce `INPUT_WIDTH` and `INPUT_HEIGHT` in Config
- Use GPU acceleration if available
- Increase `FRAME_SKIP` for real-time processing (note: affects output)

### Memory Issues
- Reduce `SPEED_CALCULATION_FRAMES`
- Lower `INPUT_WIDTH` and `INPUT_HEIGHT`
- Process smaller video segments

---

## 📡 API Endpoints

### Core Endpoints
- `GET /` - System information
- `GET /api/health` - Health check
- `GET /api/stats` - Get all statistics
- `GET /api/violations` - Get violations list

### Stream Management
- `POST /api/start-stream/{stream_id}` - Start video stream
- `POST /api/stop-stream/{stream_id}` - Stop stream
- `POST /api/stop-all-streams` - Stop all streams
- `POST /api/upload-video/{stream_id}` - Upload and process video

### Real-time Streaming
- `GET /stream/{stream_id}` - MJPEG video stream
- `WS /ws` - WebSocket for live updates

### Web Interface
- `GET /dashboard` - Main dashboard

---

## 🎨 Dashboard Features

### Overview Page
- Total vehicles detected
- Active streams status
- Real-time FPS
- Violation summary

### Surveillance Page
- Live video streams
- Multi-stream grid view
- Stream controls (start/stop)
- Upload video option

### Violations Page
- Complete violation list
- Filter by type
- Violation details
- Violation images
- Export options

---

## 📦 Project Structure

```
traffic-monitoring-system/
├── backend_complete.py          # Complete system (USE THIS)
├── requirements.txt             # Python dependencies
├── yolov8n.pt                  # YOLO model
├── README.md                   # This file
├── data/
│   ├── uploads/                # Uploaded videos
│   ├── output/                 # Processed videos
│   ├── violations/             # Violation images
│   └── logs/                   # System logs
└── frontend/
    ├── dashboard.html          # Web dashboard
    ├── styles.css             # Dashboard styles
    └── app.js                 # Dashboard logic
```

---

## 🚀 Advanced Features

### Multi-Stream Support
Process up to 4 video streams simultaneously:
```python
# Start multiple streams
requests.post('http://localhost:8000/api/start-stream/0', 
              params={'stream_url': 'video1.mp4'})
requests.post('http://localhost:8000/api/start-stream/1', 
              params={'stream_url': 'video2.mp4'})
```

### WebSocket Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'violation') {
        console.log('New violation:', data.data);
    }
    
    if (data.type === 'stats_update') {
        console.log('Stats:', data.data);
    }   
};
```

---

## 🎯 Performance Tips

### For Best Results:
1. **Use good quality videos** with clear view of vehicles
2. **Ensure traffic lights are visible** in frame
3. **Calibrate PIXELS_PER_METER** for accurate speed calculation
4. **Adjust stop line position** to match your video
5. **Configure lane count** to match your traffic scenario

### For Real-time Processing:
1. Use GPU acceleration (CUDA)
2. Reduce input resolution if needed
3. Use webcam or live stream
4. Optimize CONFIDENCE_THRESHOLD

---

## 📝 Notes

### Traffic Signal Detection
- **Automatic**: System automatically detects traffic lights in the video
- **Color Detection**: Uses HSV color space for robust detection
- **State Tracking**: Uses majority voting for stable state detection
- **No Manual Configuration Needed**: Just ensure traffic light is visible

### Speed Calculation
- Based on pixel displacement over time
- Requires calibration of `PIXELS_PER_METER` for accuracy
- Use known distances in your video for calibration
- More frames = more accurate speed estimation

### Violation Recording
- Each violation is recorded only once per vehicle
- Violation images saved automatically
- Complete violation data available via API
- WebSocket notification for real-time alerts

---

## 🐛 Known Issues & Solutions

### Issue: Output video plays as individual frames
**Solution**: ✅ Fixed in v4.0 - Now uses proper frame timing and fixed FPS

### Issue: Traffic light not detected
**Solution**: System auto-detects, but ensure light is visible in upper portion of frame

### Issue: Inaccurate speed readings
**Solution**: Calibrate `PIXELS_PER_METER` using known distances in your video

---

## 🆘 Support

For issues or questions:
1. Check the API documentation: http://localhost:8000/docs
2. Review the logs in `data/logs/`
3. Ensure all dependencies are installed
4. Verify video format is supported

---

## 📄 License

This project is for educational and monitoring purposes.

---

## 🎉 Acknowledgments

- **YOLOv8** by Ultralytics - Object detection
- **ByteTrack** - Vehicle tracking
- **FastAPI** - Web framework
- **OpenCV** - Computer vision

---

## 🔄 Version History

### v4.0.0 (Current) - Complete System
- ✅ All violation types implemented
- ✅ Smooth video output fixed
- ✅ Traffic signal auto-detection
- ✅ Lane detection
- ✅ Distance monitoring
- ✅ Complete violation recording
- ✅ High-quality video output

### v3.0.0 - Multi-stream Support
- Multi-stream processing
- WebSocket updates
- Basic violation detection

### v2.0.0 - Speed Detection
- Speed calculation
- Speed violations
- Basic tracking

### v1.0.0 - Initial Release
- Vehicle detection
- Basic tracking

---

**Made with ❤️ for Traffic Safety**
