# 🚀 Traffic Monitoring System - Deployment Guide

This guide will help you deploy your Traffic Monitoring System:
- **Backend** → Render (Docker)
- **Frontend** → Vercel (Static)

---

## 📋 Prerequisites

1. **GitHub Account** - Your code must be in a GitHub repository
2. **Render Account** - Free at [render.com](https://render.com)
3. **Vercel Account** - Free at [vercel.com](https://vercel.com)

---

## 🔧 Step 1: Push Code to GitHub

If you haven't already, push your code to GitHub:

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - Traffic Monitoring System"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push to GitHub
git push -u origin main
```

---

## 🖥️ Step 2: Deploy Backend to Render

### Option A: Using Render Blueprint (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Blueprint"**
3. Connect your GitHub repository
4. Render will auto-detect `render.yaml` and configure everything
5. Click **"Apply"**
6. Wait for deployment (5-10 minutes for first build)

### Option B: Manual Setup

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `traffic-monitoring-api`
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Runtime**: `Docker`
   - **Instance Type**: Free
5. Click **"Create Web Service"**
6. Wait for deployment (5-10 minutes)

### Get Your Backend URL

After deployment, your backend URL will be:
```
https://traffic-monitoring-api.onrender.com
```
(Or the name you chose)

**Copy this URL** - you'll need it for the frontend!

---

## 🎨 Step 3: Update Frontend with Backend URL

Before deploying frontend, update the API URL in `frontend/dashboard.html`:

Find this line (around line 1252):
```javascript
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000'
    : 'https://traffic-monitoring-api.onrender.com';  // <-- UPDATE THIS
```

Replace `traffic-monitoring-api` with your actual Render service name.

---

## 🌐 Step 4: Deploy Frontend to Vercel

1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click **"Add New..."** → **"Project"**
3. **Import** your GitHub repository
4. Configure the project:
   - **Framework Preset**: Other
   - **Root Directory**: Click "Edit" → Select `frontend`
   - **Build Command**: Leave empty
   - **Output Directory**: Leave empty (or `.`)
5. Click **"Deploy"**
6. Wait for deployment (1-2 minutes)

### Your Frontend URL

After deployment, your frontend will be at:
```
https://your-project-name.vercel.app
```

---

## ✅ Step 5: Update Backend CORS (Important!)

After getting your Vercel URL, update the CORS settings in `backend_complete.py`:

Find the CORS middleware section and add your Vercel domain:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://your-project-name.vercel.app",  # <-- ADD YOUR VERCEL URL
        "https://traffic-monitoring-api.onrender.com",
    ],
    ...
)
```

Then push the changes and Render will auto-redeploy.

---

## 🧪 Step 6: Test Your Deployment

1. **Test Backend Health**:
   - Open: `https://your-render-url.onrender.com/health`
   - Should return: `{"status": "healthy", ...}`

2. **Test Frontend**:
   - Open: `https://your-project.vercel.app`
   - Dashboard should load
   - Check browser console for API connection status

3. **Test Video Upload**:
   - Go to Monitoring page
   - Upload a test video
   - Stream should start processing

---

## 🔍 Troubleshooting

### Backend Issues

**"Service unavailable" or slow response:**
- Free tier on Render sleeps after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds to wake up
- Consider upgrading to paid tier for always-on service

**Build fails:**
- Check Render logs for errors
- Ensure all dependencies are in `requirements.txt`
- Verify `yolov8n.pt` file is committed to git

**Memory errors:**
- Free tier has limited memory (512MB)
- Video processing is memory-intensive
- Consider upgrading to Starter plan ($7/month)

### Frontend Issues

**API connection failed:**
- Verify `API_BASE_URL` in dashboard.html is correct
- Check browser console for CORS errors
- Ensure backend CORS allows your Vercel domain

**WebSocket not connecting:**
- WebSocket on free Render tier may timeout
- The frontend falls back to HTTP polling automatically

---

## 📁 Project Structure After Setup

```
KIET-AID2-Team-5/
├── backend_complete.py      # FastAPI backend
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── render.yaml             # Render deployment config
├── yolov8n.pt             # YOLO model weights
├── .dockerignore          # Docker ignore rules
├── .gitignore             # Git ignore rules
├── data/                  # Runtime data (created by app)
│   ├── violations/
│   ├── logs/
│   ├── uploads/
│   └── output/
└── frontend/
    ├── dashboard.html     # Main dashboard
    ├── vercel.json       # Vercel deployment config
    └── README.md         # Frontend readme
```

---

## 💰 Cost Summary

| Service | Plan | Cost |
|---------|------|------|
| Render Backend | Free | $0/month |
| Vercel Frontend | Hobby | $0/month |
| **Total** | | **$0/month** |

**Note**: Free tier limitations:
- Render: 750 hours/month, sleeps after 15min inactivity
- Vercel: 100GB bandwidth/month

---

## 🚀 Quick Commands Reference

```bash
# Test locally before deploying
uvicorn backend_complete:app --reload --host 0.0.0.0 --port 8000

# Build Docker image locally (optional)
docker build -t traffic-monitoring .

# Run Docker locally (optional)
docker run -p 8000:8000 traffic-monitoring

# Check Render logs (after deployment)
# Use Render dashboard → Your Service → Logs
```

---

## 📞 Support

If you encounter issues:
1. Check Render logs for backend errors
2. Check browser console for frontend errors
3. Verify all URLs are correctly configured
4. Ensure CORS settings match your domains

Good luck with your deployment! 🎉
