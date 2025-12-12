# 🚗 Traffic Monitoring System v2.0

## Advanced AI-Powered Traffic Monitoring with Speed Detection

A complete real-time traffic monitoring solution featuring vehicle detection, tracking, speed calculation, and violation detection using YOLOv8 and ByteTrack.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Real-time Vehicle Detection** - Detects cars, trucks, buses, motorcycles, and bicycles
- **Vehicle Tracking** - Persistent tracking across frames using ByteTrack algorithm
- **Speed Detection** - Real-time speed calculation in km/h
- **Speed Limit Enforcement** - Automatic detection of vehicles exceeding speed limits
- **Traffic Signal Line Detection** - Yellow line for monitoring traffic flow (not a red light)
- **License Plate Recognition** - Optional OCR-based plate reading
- **Violation Recording** - Automatic capture and storage of violation images

### 📊 Advanced Analytics
- Average speed calculation
- Vehicle classification by type
- Active vehicle tracking
- Performance metrics (FPS, processing time)
- Historical violation records

### 🌐 Modern Web Interface
- Real-time video streaming with MJPEG
- WebSocket-based live updates
- Responsive dashboard design
- Speed analytics visualization
- Violation notifications

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam or video file
- 4GB+ RAM recommended
- GPU optional (CPU works fine)

### Installation

1. **Clone or download the project**
```powershell
cd p:\projects\traffic-monitoring-system
```

python -m pip install --upgrade pip setuptools wheel
py -3.10 -m venv venv
pip install numpy==1.26.4
python --version
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
pip install -r requirements.txt
pip install streamlink imageio imageio-ffmpeg pafy
pip install --upgrade yt-dlp
yt-dlp --version    

2. **Install dependencies**
```powershell
pip install -r requirements.txt
```

3. **Run the system**
```powershell
python run.py
```

4. **Access the dashboard**
Open your browser and navigate to:
```
http://localhost:8000/dashboard
```

---

## 📖 Usage Guide

### Starting a Video Stream

1. **Using Webcam**
   - Enter `0` for default camera
   - Enter `1`, `2`, etc. for additional cameras

2. **Using Video File (URL/Path)**
   - Enter full path: `C:\Videos\traffic.mp4`
   - Or use relative path: `sample_video.mp4`

3. **Upload Video File**
   - Click the **"📁 Upload Video"** button
   - Select a video file from your computer
   - Supported formats: MP4, AVI, MOV, MKV, FLV, WMV, WebM, M4V
   - Maximum recommended size: 500MB (larger files supported but slower)
   - Video will be uploaded and processing starts automatically

4. **Using YouTube Video**
   - Enter YouTube URL: `https://www.youtube.com/watch?v=VIDEO_ID`
   - Requires `yt-dlp` to be installed

5. **Using IP Camera**
   - Enter RTSP URL: `rtsp://username:password@ip:port/stream`

### Configuration

Edit `backend_app.py` to customize:

```python
class Config:
    # Detection settings
    CONFIDENCE_THRESHOLD = 0.5  # Detection confidence (0-1)
    
    # Traffic Signal Line
    SIGNAL_LINE_POSITION = 0.6  # Line position (0-1, 0=top, 1=bottom)
    SIGNAL_LINE_COLOR = (0, 255, 255)  # Yellow (BGR format)
    
    # Speed Detection
    SPEED_LIMIT_KMH = 60  # Speed limit in km/h
    PIXELS_PER_METER = 8.0  # Calibration factor
    SPEED_CALCULATION_FRAMES = 30  # Frames for speed calculation
```

### Speed Calibration

To calibrate speed detection for your camera:

1. Measure a known distance in the camera view (e.g., 10 meters)
2. Count the pixels covering that distance
3. Calculate: `PIXELS_PER_METER = pixels / meters`
4. Update the config value

---

## 📁 Project Structure

```
traffic-monitoring-system/
├── backend_app.py              # Main FastAPI backend with AI processing
├── frontend/
│   └── index.html             # Modern responsive dashboard
├── data/
│   ├── violations/            # Violation images
│   └── logs/                  # Event logs
├── requirements.txt           # Python dependencies
├── yolov8n.pt                # YOLO model (auto-downloaded)
└── README.md                  # This file
```

---

## 🔧 API Endpoints

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/api/health` | GET | Health check |
| `/api/start-stream` | POST | Start video processing |
| `/api/upload-video` | POST | Upload video file |
| `/api/stop-stream` | POST | Stop processing |
| `/api/stats` | GET | System statistics |
| `/api/violations` | GET | Speed violations |
| `/api/vehicle-count` | GET | Vehicle counts |
| `/api/plates` | GET | Detected plates |
| `/stream` | GET | MJPEG video stream |
| `/dashboard` | GET | Web dashboard |
| `/docs` | GET | API documentation |

### WebSocket

Connect to `/ws` for real-time updates:
- Vehicle detections
- Speed violations
- System events

---

## 🎨 Features Explained

### Traffic Signal Line

The **yellow traffic signal line** is a virtual detection line placed on the video feed:
- **Purpose**: Monitor when vehicles cross a specific point
- **Not a red light**: It's a reference line for counting and tracking
- **Customizable**: Position can be adjusted via config
- **Use cases**: Traffic flow counting, intersection monitoring

### Speed Detection

Speed is calculated using:
1. **Vehicle tracking** across multiple frames
2. **Distance calculation** using pixel-to-meter conversion
3. **Time measurement** between position updates
4. **Speed formula**: Speed = Distance / Time

**Accuracy factors**:
- Camera angle (perpendicular is best)
- Calibration quality
- Frame rate
- Vehicle movement direction

### Violation Detection

When a vehicle exceeds the speed limit:
1. ⚠️ Violation is triggered
2. 📸 Frame is captured and saved
3. 📊 Data is logged (vehicle type, speed, timestamp)
4. 🔴 Real-time alert via WebSocket
5. 📝 Added to violations list

---

## 🛠️ Troubleshooting

### Video stream not loading
- Check if port 8000 is available
- Verify video source is accessible
- Try a different browser
- Check console for errors

### Slow performance
- Reduce `INPUT_WIDTH` and `INPUT_HEIGHT` in config
- Increase `FRAME_SKIP` to process fewer frames
- Use GPU if available
- Close other applications

### Speed detection inaccurate
- Calibrate `PIXELS_PER_METER` for your camera
- Ensure camera is perpendicular to traffic flow
- Adjust `SPEED_CALCULATION_FRAMES` (higher = smoother but slower response)

### Frontend not connecting to backend
- Ensure backend is running on port 8000
- Check CORS settings in `backend_app.py`
- Verify firewall isn't blocking connections
- Try accessing `http://localhost:8000/api/health`

---

## 📊 Performance Tips

### For Better FPS
```python
Config.FRAME_SKIP = 2  # Process every 2nd frame
Config.INPUT_WIDTH = 960
Config.INPUT_HEIGHT = 540
```

### For Better Accuracy
```python
Config.CONFIDENCE_THRESHOLD = 0.6  # Higher confidence
Config.FRAME_SKIP = 1  # Process every frame
Config.SPEED_CALCULATION_FRAMES = 50  # More frames for speed
```

### For Production
- Use GPU acceleration
- Implement frame queue
- Add database for long-term storage
- Deploy with nginx + gunicorn
- Use Redis for caching

---

## 🔐 Security Considerations

For production deployment:

1. **Authentication**: Add user login
2. **HTTPS**: Use SSL certificates
3. **API Keys**: Protect endpoints
4. **Rate Limiting**: Prevent abuse
5. **Data Privacy**: Secure violation images
6. **Access Control**: Role-based permissions

---

## 📚 Dependencies

Core libraries:
- **FastAPI** - Web framework
- **Uvicorn** - ASGI server
- **OpenCV** - Video processing
- **Ultralytics (YOLOv8)** - Object detection
- **PyTorch** - Deep learning backend
- **EasyOCR** - License plate reading (optional)

---

## 🤝 Contributing

Improvements welcome! Areas for contribution:
- Better speed calculation algorithms
- Database integration
- Mobile app
- Advanced analytics
- Multi-camera support
- Vehicle re-identification

---

## 📄 License

This project is for educational and research purposes.

---

## 🎯 Use Cases

- **Traffic Management**: Monitor traffic flow and congestion
- **Speed Enforcement**: Automatic speed violation detection
- **Parking Management**: Vehicle counting and tracking
- **Smart Cities**: Integration with urban infrastructure
- **Research**: Traffic pattern analysis
- **Safety**: Accident detection and prevention

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check browser console for errors
4. Verify all dependencies are installed

---

## 🚀 Advanced Features (Coming Soon)

- [ ] Multi-camera support
- [ ] Cloud storage integration
- [ ] Advanced analytics dashboard
- [ ] Mobile notifications
- [ ] AI-powered incident detection
- [ ] Traffic prediction
- [ ] Heat map visualization
- [ ] Export reports (PDF/Excel)

---

## 📝 Version History

### v2.0.0 (Current)
- ✅ Speed detection and monitoring
- ✅ Speed limit violation detection
- ✅ Traffic signal line (yellow)
- ✅ Improved vehicle tracking
- ✅ Better frontend-backend connectivity
- ✅ Real-time WebSocket updates
- ✅ Modern responsive UI
- ✅ Performance optimizations

### v1.0.0
- Basic vehicle detection
- Simple tracking
- License plate recognition
- Web dashboard

---

## 🎓 Technical Details

### Architecture
```
┌─────────────┐
│   Browser   │ ← WebSocket + HTTP
└─────────────┘
       ↕
┌─────────────┐
│  FastAPI    │ ← REST API + WebSocket
│  Backend    │
└─────────────┘
       ↕
┌─────────────┐
│  YOLOv8 +   │ ← AI Processing
│  ByteTrack  │
└─────────────┘
       ↕
┌─────────────┐
│   OpenCV    │ ← Video Processing
└─────────────┘
```

### Processing Pipeline
1. **Video Input** → Frame capture
2. **Detection** → YOLOv8 object detection
3. **Tracking** → ByteTrack multi-object tracking
4. **Speed Calc** → Distance/time calculation
5. **Violation Check** → Compare with speed limit
6. **Annotation** → Draw boxes, labels, speed
7. **Stream Output** → MJPEG encoding
8. **Web Display** → Browser rendering

---

## 💡 Tips and Tricks

1. **Best camera angle**: 30-45° from horizontal
2. **Good lighting**: Improves detection accuracy
3. **Stable mount**: Reduces false speed readings
4. **Regular calibration**: Verify PIXELS_PER_METER periodically
5. **Test with known speeds**: Validate accuracy with controlled tests

---

**Built with ❤️ using YOLOv8, FastAPI, and modern web technologies**

*For a complete, production-ready traffic monitoring system!*
