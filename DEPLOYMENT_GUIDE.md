# PTU Traffic Monitoring System - Deployment Guide

## 🚀 Complete Deployment Steps

### 📋 Prerequisites

1. **Python 3.8+** installed
2. **Git** installed
3. **YOLOv8 model file** (`yolov8n.pt`)

---

## 🔧 Step 1: Setup Project

### Clone/Download Project
```bash
# If using Git
cd P:\projects
git clone <your-repo-url> traffic-monitoring-system
cd traffic-monitoring-system

# Or if already downloaded, navigate to directory
cd P:\projects\traffic-monitoring-system
```

### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
.\venv\Scripts\activate.bat

# Linux/Mac:
source venv/bin/activate
```

---

## 📦 Step 2: Install Dependencies

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# Install yt-dlp for YouTube support (optional)
pip install yt-dlp
```

### Verify Installation
```bash
python -c "import cv2, torch, ultralytics; print('All packages installed successfully!')"
```

---

## 📥 Step 3: Download YOLOv8 Model

The system needs the YOLOv8 model file:

```bash
# Option 1: Auto-download (will download on first run)
# Just run the system, it will download automatically

# Option 2: Manual download
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

This will download `yolov8n.pt` to your project directory.

---

## 🗂️ Step 4: Create Data Directories

```bash
# Create all required directories
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/violations', 'data/logs', 'data/output', 'data/output_videos', 'data/uploads']]"
```

Or manually create:
```
data/
├── violations/
├── logs/
├── output/
├── output_videos/
└── uploads/
```

---

## ▶️ Step 5: Run the System

### Start Backend Server
```bash
# Make sure virtual environment is activated
python backend_complete.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Open Dashboard
1. Open your browser
2. Navigate to: `http://localhost:8000/dashboard`
3. You should see the PTU Traffic Monitoring System dashboard

---

## 🎥 Step 6: Add Video Streams

### Option A: Upload Video File
1. Go to **Monitoring** page
2. Click **Choose File** for any stream
3. Select a video file (.mp4, .avi)
4. Click **Upload & Start**

### Option B: Use YouTube Live Stream
1. Go to **Monitoring** page
2. Enter YouTube URL in the stream input
3. Click **Start Stream**

### Option C: Use Direct Video URL
1. Enter direct video URL (http://...)
2. Click **Start Stream**

---

## 🔍 Step 7: Monitor & View Violations

### Real-time Monitoring
- Watch live video feeds on **Monitoring** page
- See detected vehicles, speeds, and violations
- View traffic light detection, lane lines, and zebra crossings

### View Violation Records
- Go to **Violations** page
- Filter by type, stream, vehicle
- Export violation data

### Dashboard Statistics
- Total vehicles detected
- Violation counts by type
- Active surveillance streams
- Real-time signal detection status

---

## 🌐 Step 8: Deploy to Production (Optional)

### Option 1: Local Network Deployment

1. **Find your IP address:**
```bash
# Windows
ipconfig

# Linux/Mac
ifconfig
```

2. **Update backend to allow external access:**

The backend is already configured for `0.0.0.0`, so it's accessible from your network.

3. **Access from other devices:**
```
http://YOUR_IP_ADDRESS:8000/dashboard
```

Example: `http://192.168.1.100:8000/dashboard`

---

### Option 2: Cloud Deployment (Heroku, AWS, etc.)

#### Heroku Deployment

1. **Create `Procfile`:**
```bash
echo "web: uvicorn backend_complete:app --host 0.0.0.0 --port $PORT" > Procfile
```

2. **Deploy to Heroku:**
```bash
heroku login
heroku create ptu-traffic-monitor
git add .
git commit -m "Initial deployment"
git push heroku main
heroku open
```

3. **Set environment variables:**
```bash
heroku config:set ENVIRONMENT=production
```

---

#### AWS EC2 Deployment

1. **Launch EC2 instance** (Ubuntu 20.04)

2. **SSH into instance:**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Install dependencies:**
```bash
sudo apt update
sudo apt install python3-pip python3-venv nginx
```

4. **Clone project:**
```bash
git clone <your-repo> /home/ubuntu/traffic-monitoring
cd /home/ubuntu/traffic-monitoring
```

5. **Setup virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

6. **Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/traffic-monitor
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

7. **Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/traffic-monitor /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

8. **Run with systemd:**
```bash
sudo nano /etc/systemd/system/traffic-monitor.service
```

Add:
```ini
[Unit]
Description=PTU Traffic Monitoring System
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/traffic-monitoring
Environment="PATH=/home/ubuntu/traffic-monitoring/venv/bin"
ExecStart=/home/ubuntu/traffic-monitoring/venv/bin/python backend_complete.py

[Install]
WantedBy=multi-user.target
```

9. **Start service:**
```bash
sudo systemctl start traffic-monitor
sudo systemctl enable traffic-monitor
sudo systemctl status traffic-monitor
```

---

### Option 3: Docker Deployment

1. **Create `Dockerfile`:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "backend_complete.py"]
```

2. **Create `docker-compose.yml`:**
```yaml
version: '3.8'

services:
  traffic-monitor:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

3. **Build and run:**
```bash
docker-compose build
docker-compose up -d
```

---

## 🔒 Step 9: Security (Production)

1. **Set up HTTPS** (use Let's Encrypt):
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

2. **Enable firewall:**
```bash
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 22
sudo ufw enable
```

3. **Update CORS settings** in `backend_complete.py` for production:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Change from ["*"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 Step 10: Monitoring & Maintenance

### View Logs
```bash
# Application logs
tail -f data/logs/app.log

# System service logs (if using systemd)
sudo journalctl -u traffic-monitor -f
```

### Stop System
```bash
# Ctrl+C if running in terminal

# If running as service
sudo systemctl stop traffic-monitor
```

### Update System
```bash
git pull
pip install -r requirements.txt
sudo systemctl restart traffic-monitor
```

---

## 🆘 Troubleshooting

### Issue: Port already in use
```bash
# Windows - Kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <process_id> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Issue: CUDA/GPU not detected
```bash
# Install CPU version of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: OpenCV errors
```bash
# Reinstall opencv
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Issue: YOLOv8 model not found
```bash
# Download manually
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## ✅ Verify Deployment

1. ✅ Backend running on http://localhost:8000
2. ✅ Dashboard accessible at http://localhost:8000/dashboard
3. ✅ Can upload and process video files
4. ✅ Real-time violation detection working
5. ✅ WebSocket updates showing live stats
6. ✅ Violation images saved to `data/violations/`

---

## 📞 Support

For issues or questions:
- Check logs in `data/logs/`
- Review error messages in terminal
- Verify all dependencies installed correctly

---

## 🎉 You're All Set!

Your PTU Traffic Monitoring System is now deployed and ready to monitor traffic violations in real-time!

**Quick Start Commands:**
```bash
# Activate environment
.\venv\Scripts\Activate.ps1

# Start system
python backend_complete.py

# Open dashboard
# Navigate to: http://localhost:8000/dashboard
```
