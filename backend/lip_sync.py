import subprocess
import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# FFmpeg absolute path
FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def make_video(audio_path, session_id):
    """Create lip-synced video using animated avatar"""
    try:
        logger.info(f"=== VIDEO CREATION START for session {session_id} ===")
        logger.info(f"Input audio file: {audio_path}")
        
        # Verify audio file
        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            return None
            
        audio_size = os.path.getsize(audio_path)
        logger.info(f"Audio file size: {audio_size} bytes")
        
        if audio_size < 1000:
            logger.error(f"Audio file too small: {audio_size} bytes")
            return None
        
        # Use animated video with the anime-style avatar
        logger.info("Creating anime-style avatar video")
        return create_animated_video(audio_path, session_id)
        
    except Exception as e:
        logger.error(f"Error in video generation: {str(e)}")
        return create_basic_video(audio_path, session_id)

def create_animated_video(audio_path, session_id):
    """Create animated avatar video with mouth movement synchronized to audio"""
    try:
        logger.info("=== ANIMATED VIDEO CREATION START ===")
        
        import librosa
        
        # Load and analyze audio
        audio, sr = librosa.load(audio_path)
        duration = len(audio) / sr
        logger.info(f"Audio duration: {duration:.2f} seconds")
        
        # Create anime-style avatar
        avatar_path = f"temp/{session_id}_avatar.jpg"
        create_anime_female_avatar(avatar_path)
        
        output_path = f"temp/{session_id}_animated_video.mp4"
        
        # Analyze audio for mouth animation
        hop_length = 512
        frame_length = 2048
        
        # Get audio features for lip sync
        stft = librosa.stft(audio, hop_length=hop_length, n_fft=frame_length)
        magnitude = np.abs(stft)
        
        # Get RMS energy for mouth opening
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Get spectral centroid for mouth shape
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        
        # Smooth the features
        from scipy import ndimage
        rms_smooth = ndimage.gaussian_filter1d(rms, sigma=1.0)
        centroid_smooth = ndimage.gaussian_filter1d(spectral_centroid, sigma=1.0)
        
        # Create video frames
        img = cv2.imread(avatar_path)
        height, width = img.shape[:2]
        
        fps = 25
        total_frames = int(duration * fps)
        
        logger.info(f"Creating {total_frames} frames at {fps} FPS")
        
        # Create video without audio first
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = f"temp/{session_id}_video_no_audio.mp4"
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        for frame_num in range(total_frames):
            frame = img.copy()
            
            # Calculate audio features for this frame
            time_sec = frame_num / fps
            feature_index = int(time_sec * sr / hop_length)
            
            if feature_index < len(rms_smooth):
                # Mouth opening based on RMS energy
                mouth_opening = min(rms_smooth[feature_index] * 6, 1.0)
                
                # Mouth shape based on spectral centroid
                centroid_val = centroid_smooth[feature_index] if feature_index < len(centroid_smooth) else 1000
                mouth_width = 1.0 + (centroid_val - 1000) / 4000
                mouth_width = max(0.8, min(mouth_width, 1.2))
            else:
                mouth_opening = 0
                mouth_width = 1.0
            
            # Animate mouth with calculated parameters
            animate_anime_mouth(frame, mouth_opening, mouth_width)
            
            # Add blinking animation
            if frame_num % 80 < 3:  # Blink every ~3 seconds
                add_anime_blink_effect(frame)
            
            # Add subtle head movement
            if frame_num % 60 < 30:  # Gentle movement
                head_offset = int(np.sin(frame_num * 0.1) * 1.5)
                frame = shift_frame(frame, 0, head_offset)
            
            out.write(frame)
        
        out.release()
        
        logger.info("Video frames created, adding audio")
        
        # Add audio to video using FFmpeg
        add_audio_to_video(temp_video_path, audio_path, output_path)
        
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Verify output
        if os.path.exists(output_path) and test_video_has_audio(output_path):
            logger.info("=== ANIMATED VIDEO CREATION SUCCESS ===")
            return output_path
        else:
            logger.error("Animated video creation failed or has no audio")
            return create_basic_video(audio_path, session_id)
        
    except Exception as e:
        logger.error(f"Error creating animated video: {str(e)}")
        return create_basic_video(audio_path, session_id)

def create_anime_female_avatar(output_path):
    """Create anime-style female avatar similar to the reference image"""
    try:
        # Create high resolution canvas
        img = np.ones((720, 720, 3), dtype=np.uint8)
        
        # Professional news studio background (warm, soft lighting)
        for y in range(720):
            factor = y / 720
            # Warm gradient background
            img[y, :] = [
                int(200 + factor * 35),   # Blue
                int(210 + factor * 35),   # Green  
                int(220 + factor * 35)    # Red
            ]
        
        # Add subtle lighting effect
        center_light = (360, 250)
        for y in range(720):
            for x in range(720):
                distance = np.sqrt((x - center_light[0])**2 + (y - center_light[1])**2)
                if distance < 300:
                    light_factor = (300 - distance) / 300 * 0.15
                    img[y, x] = np.clip(img[y, x] + img[y, x] * light_factor, 0, 255)
        
        # Character positioning
        face_center_x, face_center_y = 360, 320
        
        # Body/Clothing (professional but cute)
        # Shoulders
        shoulder_points = np.array([
            [200, 550], [520, 550], [580, 720], [140, 720]
        ], np.int32)
        cv2.fillPoly(img, [shoulder_points], (240, 240, 245))  # Light colored outfit
        
        # Collar/shirt details
        collar_points = np.array([
            [300, 520], [420, 520], [410, 580], [310, 580]
        ], np.int32)
        cv2.fillPoly(img, [collar_points], (255, 255, 255))  # White shirt/blouse
        
        # Neck
        cv2.ellipse(img, (face_center_x, face_center_y + 110), (25, 40), 0, 0, 360, (255, 220, 200), -1)
        
        # Face shape (anime-style, rounder)
        face_width, face_height = 120, 130
        cv2.ellipse(img, (face_center_x, face_center_y), (face_width, face_height), 
                   0, 0, 360, (255, 230, 210), -1)  # Anime skin tone
        
        # Face shading (minimal, anime-style)
        cv2.ellipse(img, (face_center_x, face_center_y + 20), (100, 110), 
                   0, 0, 360, (250, 220, 195), -1)
        
        # Hair (blonde, similar to reference)
        # Main hair mass
        hair_points = np.array([
            [face_center_x - 140, face_center_y - 120],
            [face_center_x + 140, face_center_y - 120],
            [face_center_x + 130, face_center_y + 80],
            [face_center_x + 100, face_center_y + 120],
            [face_center_x - 100, face_center_y + 120],
            [face_center_x - 130, face_center_y + 80]
        ], np.int32)
        cv2.fillPoly(img, [hair_points], (255, 235, 160))  # Blonde hair
        
        # Hair highlights and strands
        cv2.ellipse(img, (face_center_x - 60, face_center_y - 80), (40, 60), -30, 0, 360, (255, 245, 180), -1)
        cv2.ellipse(img, (face_center_x + 60, face_center_y - 80), (40, 60), 30, 0, 360, (255, 245, 180), -1)
        cv2.ellipse(img, (face_center_x, face_center_y - 100), (50, 40), 0, 0, 360, (255, 245, 180), -1)
        
        # Hair shadows
        cv2.ellipse(img, (face_center_x - 70, face_center_y - 60), (30, 50), -20, 0, 360, (240, 210, 140), -1)
        cv2.ellipse(img, (face_center_x + 70, face_center_y - 60), (30, 50), 20, 0, 360, (240, 210, 140), -1)
        
        # Twin tails/side hair (anime style)
        left_tail = (face_center_x - 110, face_center_y + 30)
        right_tail = (face_center_x + 110, face_center_y + 30)
        cv2.ellipse(img, left_tail, (35, 80), -15, 0, 360, (255, 235, 160), -1)
        cv2.ellipse(img, right_tail, (35, 80), 15, 0, 360, (255, 235, 160), -1)
        
        # Eyes (large anime-style, blue like reference)
        eye_y = face_center_y - 30
        left_eye_x, right_eye_x = face_center_x - 45, face_center_x + 45
        
        # Eye whites (larger for anime style)
        cv2.ellipse(img, (left_eye_x, eye_y), (28, 20), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (right_eye_x, eye_y), (28, 20), 0, 0, 360, (255, 255, 255), -1)
        
        # Iris (bright blue like reference)
        cv2.circle(img, (left_eye_x, eye_y + 2), 18, (100, 150, 255), -1)
        cv2.circle(img, (right_eye_x, eye_y + 2), 18, (100, 150, 255), -1)
        
        # Inner iris details
        cv2.circle(img, (left_eye_x, eye_y + 2), 12, (60, 120, 255), -1)
        cv2.circle(img, (right_eye_x, eye_y + 2), 12, (60, 120, 255), -1)
        
        # Pupils
        cv2.circle(img, (left_eye_x, eye_y + 2), 6, (20, 20, 50), -1)
        cv2.circle(img, (right_eye_x, eye_y + 2), 6, (20, 20, 50), -1)
        
        # Eye highlights (anime sparkle)
        cv2.circle(img, (left_eye_x - 8, eye_y - 5), 4, (255, 255, 255), -1)
        cv2.circle(img, (right_eye_x - 8, eye_y - 5), 4, (255, 255, 255), -1)
        cv2.circle(img, (left_eye_x + 6, eye_y - 2), 2, (255, 255, 255), -1)
        cv2.circle(img, (right_eye_x + 6, eye_y - 2), 2, (255, 255, 255), -1)
        
        # Eyelashes (anime style)
        cv2.ellipse(img, (left_eye_x, eye_y - 15), (30, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (right_eye_x, eye_y - 15), (30, 8), 0, 0, 360, (40, 40, 40), -1)
        
        # Eyebrows (thin, anime style)
        cv2.ellipse(img, (left_eye_x - 5, eye_y - 35), (25, 4), -10, 0, 360, (200, 180, 120), -1)
        cv2.ellipse(img, (right_eye_x + 5, eye_y - 35), (25, 4), 10, 0, 360, (200, 180, 120), -1)
        
        # Small nose (anime style)
        nose_y = face_center_y + 10
        cv2.circle(img, (face_center_x - 3, nose_y), 2, (245, 210, 190), -1)
        cv2.circle(img, (face_center_x + 3, nose_y), 2, (245, 210, 190), -1)
        
        # Mouth area (will be animated)
        mouth_y = face_center_y + 35
        cv2.ellipse(img, (face_center_x, mouth_y), (20, 8), 0, 0, 360, (255, 150, 150), -1)
        
        # Cheek blush (anime style)
        cv2.ellipse(img, (face_center_x - 70, face_center_y + 15), (15, 10), 0, 0, 360, (255, 200, 200), -1)
        cv2.ellipse(img, (face_center_x + 70, face_center_y + 15), (15, 10), 0, 0, 360, (255, 200, 200), -1)
        
        # Hair accessories (small ribbons/clips like reference)
        # Left side ribbon
        ribbon_left = (face_center_x - 90, face_center_y - 40)
        cv2.ellipse(img, ribbon_left, (12, 8), 0, 0, 360, (150, 100, 255), -1)
        cv2.ellipse(img, ribbon_left, (8, 5), 0, 0, 360, (180, 130, 255), -1)
        
        # Right side ribbon
        ribbon_right = (face_center_x + 90, face_center_y - 40)
        cv2.ellipse(img, ribbon_right, (12, 8), 0, 0, 360, (150, 100, 255), -1)
        cv2.ellipse(img, ribbon_right, (8, 5), 0, 0, 360, (180, 130, 255), -1)
        
        # Apply slight anime-style smoothing
        img = cv2.bilateralFilter(img, 5, 80, 80)
        
        cv2.imwrite(output_path, img)
        logger.info(f"Anime-style female avatar created: {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating anime avatar: {str(e)}")
        raise e

def animate_anime_mouth(frame, mouth_opening, mouth_width):
    """Anime-style mouth animation"""
    try:
        mouth_center = (360, 355)  # Position for 720p anime avatar
        
        # Clear existing mouth area
        cv2.ellipse(frame, mouth_center, (25, 15), 0, 0, 360, (255, 230, 210), -1)
        
        # Calculate mouth dimensions
        base_width = int(20 * mouth_width)
        mouth_height = int(6 + mouth_opening * 12)
        
        # Clamp values
        base_width = max(12, min(base_width, 30))
        mouth_height = max(3, min(mouth_height, 18))
        
        # Draw mouth based on opening (anime style)
        if mouth_opening < 0.1:
            # Small closed mouth
            cv2.ellipse(frame, mouth_center, (base_width, 3), 0, 0, 360, (255, 120, 120), -1)
        elif mouth_opening < 0.3:
            # Slightly open
            cv2.ellipse(frame, mouth_center, (base_width, mouth_height), 0, 0, 360, (255, 120, 120), -1)
            # Small opening
            cv2.ellipse(frame, mouth_center, (base_width-4, max(1, mouth_height-3)), 0, 0, 360, (200, 50, 50), -1)
        else:
            # More open (anime surprise/speaking)
            cv2.ellipse(frame, mouth_center, (base_width, mouth_height), 0, 0, 360, (200, 50, 50), -1)
            
            # Teeth for more open mouth
            if mouth_height > 10:
                cv2.ellipse(frame, (mouth_center[0], mouth_center[1] - 2), (base_width-6, 4), 0, 0, 360, (255, 255, 255), -1)
            
            # Tongue for very open
            if mouth_height > 14:
                cv2.ellipse(frame, (mouth_center[0], mouth_center[1] + 3), (base_width-8, 3), 0, 0, 360, (255, 180, 180), -1)
        
        # Small highlight (anime style)
        if mouth_opening > 0.1:
            cv2.ellipse(frame, (mouth_center[0], mouth_center[1] - mouth_height//2), (base_width//3, 1), 0, 0, 360, (255, 200, 200), -1)
        
    except Exception as e:
        logger.error(f"Error animating anime mouth: {str(e)}")

def add_anime_blink_effect(frame):
    """Add anime-style blinking effect"""
    try:
        # Eye positions for 720p anime avatar
        left_eye_center = (315, 290)
        right_eye_center = (405, 290)
        
        # Draw closed eyes (anime style)
        cv2.ellipse(frame, left_eye_center, (28, 3), 0, 0, 360, (240, 210, 190), -1)
        cv2.ellipse(frame, right_eye_center, (28, 3), 0, 0, 360, (240, 210, 190), -1)
        
        # Eyelashes during blink
        cv2.ellipse(frame, left_eye_center, (30, 2), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(frame, right_eye_center, (30, 2), 0, 0, 360, (40, 40, 40), -1)
        
    except Exception as e:
        logger.error(f"Error adding anime blink effect: {str(e)}")

def shift_frame(frame, dx, dy):
    """Shift frame slightly for head movement"""
    try:
        rows, cols = frame.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    except:
        return frame

def add_audio_to_video(video_path, audio_path, output_path):
    """Add audio to video using FFmpeg"""
    try:
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            raise Exception("Video file not found")
            
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise Exception("Audio file not found")
        
        logger.info(f"Adding audio {audio_path} to video {video_path}")
        
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Audio added successfully to video: {output_path}")
        else:
            logger.error(f"Error adding audio - FFmpeg stderr: {result.stderr}")
            raise Exception("Failed to add audio to video")
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg process timed out in add_audio_to_video")
        raise Exception("FFmpeg timeout")
    except Exception as e:
        logger.error(f"Error in add_audio_to_video: {str(e)}")
        raise e

def test_video_has_audio(video_path):
    """Test if video has audio stream"""
    try:
        cmd = [
            FFMPEG_PATH.replace('ffmpeg.exe', 'ffprobe.exe'),
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            import json
            probe_data = json.loads(result.stdout)
            streams = probe_data.get('streams', [])
            
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    return True
            return False
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error testing video audio: {str(e)}")
        return False

def create_basic_video(audio_path, session_id):
    """Create basic video with static anime avatar"""
    try:
        logger.info("=== BASIC VIDEO CREATION START ===")
        
        avatar_path = f"temp/{session_id}_avatar.jpg"
        create_anime_female_avatar(avatar_path)
        
        output_path = f"temp/{session_id}_basic_video.mp4"
        
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return None
            
        audio_size = os.path.getsize(audio_path)
        logger.info(f"Creating basic video with audio: {audio_path} ({audio_size} bytes)")
        
        cmd = [
            FFMPEG_PATH, "-y",
            "-loop", "1", "-i", avatar_path,
            "-i", audio_path,
            "-c:v", "libx264", "-c:a", "aac",
            "-b:a", "192k", "-b:v", "2M",
            "-pix_fmt", "yuv420p",
            "-shortest", "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            logger.info(f"Basic video created: {output_path} ({output_size} bytes)")
            
            if test_video_has_audio(output_path):
                logger.info("=== BASIC VIDEO CREATION SUCCESS ===")
                return output_path
            else:
                logger.error("Basic video has no audio")
                return None
        else:
            logger.error(f"Basic video creation failed: {result.stderr}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating basic video: {str(e)}")
        return None
