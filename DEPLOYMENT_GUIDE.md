# 🚀 Deployment Guide - YouTube & Video Upload Support

## ✅ What's Already Implemented

Your system **NOW SUPPORTS BOTH**:
1. ✅ **YouTube URLs** - Paste any YouTube URL and it will work
2. ✅ **Video File Upload** - Upload MP4, AVI, MOV, MKV, etc.

## 📋 What Will Happen Now

### For YouTube URLs:
1. User enters YouTube URL in the input field (e.g., `https://youtu.be/wqctLW0Hb_0`)
2. Clicks "Start" button
3. Backend automatically:
   - Detects it's a YouTube URL
   - Tries multiple extraction methods (iOS, Android, Web, TV, Mobile)
   - Extracts the actual video stream URL
   - Starts processing
   - Saves metadata to database

### For Video Uploads:
1. User clicks "Upload Video File" or drags & drops a file
2. Clicks "Upload" button
3. Backend automatically:
   - Validates file type and size (max 500MB)
   - Saves file to `data/uploads/` directory
   - Starts processing
   - Saves metadata to database

## 🔧 Do You Need to Change Any Files?

### ✅ **NO CHANGES NEEDED** - Everything is already set up!

The frontend already has:
- ✅ YouTube URL input fields
- ✅ File upload buttons
- ✅ Both API calls implemented

The backend already has:
- ✅ `/api/start-stream/{stream_id}` endpoint (for YouTube URLs)
- ✅ `/api/upload-video/{stream_id}` endpoint (for file uploads)
- ✅ Database storage for both
- ✅ Multiple YouTube extraction methods

## 🚀 Commands to Use

### For Local Testing:
```bash
# Just run the backend (no special commands needed)
python backend_complete.py
```

### For Render Deployment:
1. **No special commands needed** - just deploy as usual
2. **Optional (for better YouTube success rate):**
   - Set environment variable in Render dashboard:
     ```
     YOUTUBE_COOKIES_FILE=/path/to/cookies.txt
     ```
   - Or export cookies using:
     ```bash
     yt-dlp --cookies-from-browser chrome --output cookies.txt
     ```

## 📝 How to Use

### Method 1: YouTube URL
1. Go to Monitoring page
2. Enter YouTube URL in "Live / YouTube Stream URL" field
   - Example: `https://youtu.be/wqctLW0Hb_0`
   - Or: `https://www.youtube.com/watch?v=VIDEO_ID`
3. Click "Start" button
4. Wait for extraction (may take 10-30 seconds)
5. Stream will start automatically

### Method 2: Video Upload
1. Go to Monitoring page
2. Click "Upload Video File" area or drag & drop
3. Select your video file (MP4, AVI, MOV, MKV, etc.)
4. Click "Upload" button
5. File uploads and processing starts automatically

## 🗄️ Database Storage

Both methods automatically save to MongoDB:
- **Stream metadata**: Source type, URLs, file paths, timestamps
- **Violation tracking**: All violations are saved
- **Stream history**: View at `/api/streams/history`

## ⚠️ Important Notes

### YouTube Extraction:
- **First attempt**: May take 10-30 seconds (tries multiple methods)
- **If it fails**: System automatically retries with different methods
- **Best for production**: Use cookies file (see above)
- **Error messages**: Will show helpful guidance if extraction fails

### File Uploads:
- **Max file size**: 500MB
- **Supported formats**: MP4, AVI, MOV, MKV, FLV, WMV, WEBM, M4V
- **Storage**: Files saved in `data/uploads/` directory
- **Cleanup**: Files are kept for processing history

## 🧪 Testing

### Test YouTube URL:
```bash
curl -X POST "http://localhost:8000/api/start-stream/0?stream_url=https://youtu.be/wqctLW0Hb_0"
```

### Test File Upload:
```bash
curl -X POST "http://localhost:8000/api/upload-video/0" \
  -F "file=@your_video.mp4"
```

## 📊 Monitoring

Check stream status:
```bash
# Get stream metadata
GET /api/stream-metadata/{stream_id}

# Get all stream history
GET /api/streams/history?limit=20
```

## 🎯 Summary

**You don't need to:**
- ❌ Change any files
- ❌ Run special commands
- ❌ Modify frontend code
- ❌ Update API endpoints

**Everything is ready to use!** Just:
1. Deploy to Render (or run locally)
2. Use YouTube URLs or upload videos
3. Everything works automatically! 🎉
