# Traffic Monitoring System (TMS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.220-orange.svg)](https://github.com/ultralytics/ultralytics)

An advanced AI-powered traffic monitoring system using YOLOv8 for real-time vehicle detection, speed estimation, and comprehensive violation tracking. Built with FastAPI backend, MongoDB database, and modern web frontend.

## 🚀 Features

### Core Functionality
- **Real-time Vehicle Detection**: YOLOv8-based object detection for accurate vehicle tracking
- **Multi-stream Processing**: Support for up to 4 concurrent video streams
- **Speed Estimation**: Advanced computer vision algorithms for vehicle speed calculation
- **Violation Detection**: Comprehensive violation types including:
  - Speed violations
  - Red light violations
  - Stop-line violations
  - Lane violations
  - Unsafe following distance
  - Wrong-way driving

### Advanced Features
- **Traffic Light Detection**: Automatic detection of red/yellow/green signals
- **Lane Detection**: Real-time lane boundary identification
- **Zebra Crossing Detection**: Pedestrian crossing monitoring
- **WebSocket Integration**: Real-time updates and notifications
- **MongoDB Storage**: Persistent violation records and user data
- **JWT Authentication**: Secure user authentication and authorization
- **Responsive Dashboard**: Modern web interface with analytics and monitoring

### Technical Stack
- **Backend**: FastAPI, Uvicorn, Python 3.8+
- **AI/ML**: YOLOv8, OpenCV, NumPy, PyTorch
- **Database**: MongoDB with Motor (async driver)
- **Frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js
- **Authentication**: JWT tokens, bcrypt password hashing
- **Deployment**: Docker, Docker Compose, Vercel (frontend), Render (backend)

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 5GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

### Software Dependencies
- **Python**: 3.8 or higher
- **MongoDB**: 4.4 or higher (local or cloud)
- **Node.js**: 16+ (for frontend deployment)
- **Docker**: 20.10+ (optional, for containerized deployment)

### Python Packages
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
websockets==12.0
python-dotenv==1.0.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
pillow==10.1.0
ultralytics==8.0.220
lapx>=0.5.2
aiofiles==23.2.1
requests==2.31.0
streamlink==6.2.0
torch==2.1.1
torchvision==0.16.1
motor==3.3.2
pymongo==4.6.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
bcrypt==4.1.2
pydantic[email]==2.5.2
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/traffic-monitoring-system.git
cd traffic-monitoring-system
```

### 2. Backend Setup

#### Option A: Using Python Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv tms_env

# Activate virtual environment
# Windows:
tms_env\Scripts\activate
# macOS/Linux:
source tms_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Docker (Alternative)
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### 3. Database Setup

#### Local MongoDB Installation
```bash
# Install MongoDB (Ubuntu/Debian)
sudo apt-get install mongodb

# Install MongoDB (macOS with Homebrew)
brew install mongodb-community
brew services start mongodb-community

# Install MongoDB (Windows)
# Download from https://www.mongodb.com/try/download/community
# Follow installation instructions
```

#### MongoDB as a Service
```bash
# Using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest

# Or use MongoDB Atlas (cloud)
# Create account at https://www.mongodb.com/atlas
# Get connection string and update environment variables
```

### 4. Environment Configuration

Create a `.env` file in the root directory (copy from `.env.example`):

```bash
cp .env.example .env
```

**Important:** Update the following values in your `.env` file:
- `SECRET_KEY`: Change to a secure random string for production
- `MONGODB_URL`: Update if using MongoDB Atlas or custom MongoDB instance
- `PRODUCTION_API_BASE_URL`: Update with your Render app URL when deployed
- `PRODUCTION_WS_BASE_URL`: Update with your Render WebSocket URL when deployed

Example `.env` configuration:
```env
# Backend
SECRET_KEY=your-unique-secret-key-here
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=traffic_monitoring

# Frontend (Production URLs - update when deployed)
PRODUCTION_API_BASE_URL=https://your-app-name.onrender.com
PRODUCTION_WS_BASE_URL=wss://your-app-name.onrender.com
```

The frontend automatically loads configuration from the backend's `/api/config` endpoint.

### 5. Download YOLOv8 Model

The system includes the YOLOv8 nano model (`yolov8n.pt`). If you need a different model:
```bash
# Download custom model (optional)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

## 🚀 Running the Application

### Development Mode

#### Start Backend Server
```bash
# From project root
python backend_complete.py

# Or with uvicorn directly
uvicorn backend_complete:app --host 0.0.0.0 --port 8000 --reload
```

#### Access the Application
- **Landing Page**: http://localhost:8000/
- **Dashboard**: http://localhost:8000/dashboard
- **Analytics**: http://localhost:8000/analytics
- **Monitoring**: http://localhost:8000/monitoring
- **API Documentation**: http://localhost:8000/docs

### Production Deployment

#### Backend Deployment (Render)
1. Create account at https://render.com
2. Connect GitHub repository
3. Create new Web Service
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `python backend_complete.py`
6. Add environment variables
7. Deploy

#### Frontend Deployment (Vercel)
1. Create account at https://vercel.com
2. Connect GitHub repository
3. Deploy the `frontend/` directory
4. Update API URLs in `script.js` for production

#### Docker Deployment
```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📖 Usage

### User Authentication
1. Visit the landing page
2. Click "Get Started" to sign up or login
3. After authentication, access the dashboard

### Stream Management
1. Navigate to the Monitoring page
2. Enter RTSP/HTTP stream URLs or upload video files
3. Click "Start Stream" for each stream
4. Monitor real-time processing and violations

### Analytics & Reporting
1. Visit the Analytics page
2. View real-time statistics and charts
3. Filter violations by date range
4. Export violation reports

### API Usage

#### Authentication Endpoints
```bash
# Signup
curl -X POST "http://localhost:8000/api/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"password123","full_name":"Test User"}'

# Login
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

#### Stream Management
```bash
# Start stream
curl -X POST "http://localhost:8000/api/start-stream/0?stream_url=rtsp://example.com/stream" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"

# Stop stream
curl -X POST "http://localhost:8000/api/stop-stream/0" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### Statistics
```bash
# Get system stats
curl "http://localhost:8000/api/stats"

# Get violations
curl "http://localhost:8000/api/db/violations?limit=100" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🔧 Configuration

### Stream Configuration
- **Max Streams**: 4 concurrent streams
- **Poll Interval**: 2 seconds for stats, 1 second for frames
- **Timeout**: 5 seconds for frame requests

### Violation Thresholds
- **Speed Limit**: Configurable per stream
- **Distance Threshold**: 2.0 meters for unsafe following
- **Detection Confidence**: 0.5 minimum for YOLOv8

### Database Collections
- `users`: User authentication data
- `violations`: Violation records with metadata
- `streams`: Stream configuration and status

## 🧪 Testing

### Backend Testing
```bash
# Run with test database
export DATABASE_NAME=traffic_monitoring_test
python backend_complete.py

# API testing with curl or Postman
# Use the /docs endpoint for interactive API testing
```

### Frontend Testing
```bash
# Serve frontend files (if not using backend static files)
cd frontend
python -m http.server 3000

# Access at http://localhost:3000
```

## 📊 Monitoring & Logs

### Application Logs
```bash
# View application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f backend
```

### Performance Monitoring
- **System Stats**: Available at `/api/stats`
- **Stream Status**: Check individual stream endpoints
- **Violation Counts**: Real-time updates via WebSocket

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use meaningful commit messages
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the excellent object detection model
- **FastAPI**: For the modern, fast web framework
- **OpenCV**: For computer vision capabilities
- **MongoDB**: For reliable document database storage

## 📞 Support

For support, email support@tms.com or create an issue in the GitHub repository.

## 🔄 Version History

- **v4.0.0**: Complete rewrite with advanced violation detection
- **v3.0.0**: Multi-stream support and MongoDB integration
- **v2.0.0**: JWT authentication and user management
- **v1.0.0**: Basic YOLOv8 integration

---

**Built with ❤️ for safer and smarter traffic management**