import subprocess
import os
import logging
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# FFmpeg absolute path
FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg-7.1.1-essentials_build\bin\ffmpeg.exe"

def add_graphics(video_path, topic, session_id):
    """Add news graphics and branding to video"""
    try:
        logger.info(f"=== GRAPHICS PROCESSING START ===")
        logger.info(f"Input video: {video_path}")
        
        # Verify input video exists and has audio
        if not os.path.exists(video_path):
            logger.error(f"Input video not found: {video_path}")
            return None
        
        # Test input video
        input_has_audio = test_video_streams(video_path, "INPUT")
        
        # Create graphics elements
        lower_third_path = create_lower_third_graphic(topic, session_id)
        logo_path = create_news_logo(session_id)

        output_path = f"outputs/{session_id}_final_video.mp4"
        os.makedirs("outputs", exist_ok=True)

        # Apply graphics with proper audio preservation
        success = apply_video_graphics(video_path, lower_third_path, logo_path, output_path)

        if success:
            # Test output video
            output_has_audio = test_video_streams(output_path, "OUTPUT")
            
            if input_has_audio and not output_has_audio:
                logger.error("*** AUDIO LOST DURING GRAPHICS PROCESSING ***")
                # Fall back to simple copy
                return simple_copy_with_audio(video_path, output_path)
            
            logger.info(f"Final video created with graphics: {output_path}")
            return output_path
        else:
            # If graphics fail, copy original video preserving audio
            return simple_copy_with_audio(video_path, output_path)

    except Exception as e:
        logger.error(f"Error adding graphics: {str(e)}")
        return simple_copy_with_audio(video_path, f"outputs/{session_id}_final_video.mp4")

def simple_copy_with_audio(input_path, output_path):
    """Simple copy preserving all streams"""
    try:
        logger.warning("Copying video without graphics to preserve audio")
        
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", input_path,
            "-c", "copy",  # Copy all streams without re-encoding
            "-movflags", "+faststart",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(output_path):
            logger.info(f"Video copied successfully: {output_path}")
            return output_path
        else:
            logger.error(f"Failed to copy video: {result.stderr}")
            return input_path
            
    except Exception as e:
        logger.error(f"Error copying video: {str(e)}")
        return input_path

def create_lower_third_graphic(topic, session_id):
    """Create professional lower third graphic"""
    try:
        width, height = 1280, 120
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Background with transparency
        draw.rectangle([0, 0, width, height], fill=(30, 30, 70, 200))
        
        # Accent stripe
        draw.rectangle([0, 0, width, 8], fill=(220, 38, 127, 255))

        # Try to load fonts
        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_small = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                font_large = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
                font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
            except:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

        # Main text
        draw.text((20, 25), "AI NEWS REPORTER", font=font_large, fill=(255, 255, 255, 255))
        
        # Topic text (truncate if too long)
        topic_text = topic.upper()[:50] + ("..." if len(topic) > 50 else "")
        draw.text((20, 65), topic_text, font=font_small, fill=(255, 255, 255, 200))

        # Live indicator
        draw.rectangle([width-120, 20, width-20, 50], fill=(220, 38, 127, 255))
        draw.text((width-110, 25), "● LIVE", font=font_small, fill=(255, 255, 255, 255))

        output_path = f"temp/{session_id}_lower_third.png"
        img.save(output_path, "PNG")
        
        logger.info(f"Lower third graphic created: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating lower third: {str(e)}")
        return None

def create_news_logo(session_id):
    """Create news logo/watermark"""
    try:
        img = Image.new('RGBA', (150, 50), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            except:
                font = ImageFont.load_default()

        # Logo background
        draw.rectangle([0, 0, 150, 50], fill=(30, 30, 70, 180))
        draw.rectangle([0, 0, 150, 3], fill=(220, 38, 127, 255))

        # Logo text
        draw.text((10, 15), "AI NEWS", font=font, fill=(255, 255, 255, 255))

        output_path = f"temp/{session_id}_logo.png"
        img.save(output_path, "PNG")
        
        logger.info(f"News logo created: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating logo: {str(e)}")
        return None

def apply_video_graphics(video_path, lower_third_path, logo_path, output_path):
    """Apply graphics to video using FFmpeg with audio preservation"""
    try:
        filter_parts = []
        input_count = 1  # Start with video input

        # Build filter chain while preserving audio
        current_video = "0:v"
        
        # Add lower third overlay
        if lower_third_path and os.path.exists(lower_third_path):
            filter_parts.append(f"[{current_video}][{input_count}:v]overlay=0:H-h-20:enable='between(t,5,1000)'[v{input_count}]")
            current_video = f"v{input_count}"
            input_count += 1

        # Add logo overlay
        if logo_path and os.path.exists(logo_path):
            filter_parts.append(f"[{current_video}][{input_count}:v]overlay=W-w-20:20:enable='between(t,0,1000)'[final]")
            final_output = "final"
        else:
            final_output = current_video

        # Build FFmpeg command
        cmd = [FFMPEG_PATH, "-y", "-i", video_path]

        # Add graphic inputs
        if lower_third_path and os.path.exists(lower_third_path):
            cmd.extend(["-i", lower_third_path])
        if logo_path and os.path.exists(logo_path):
            cmd.extend(["-i", logo_path])

        # Add filter complex if we have graphics
        if filter_parts:
            filter_complex = ";".join(filter_parts)
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", f"[{final_output}]",  # Map video output
                "-map", "0:a",  # EXPLICITLY map original audio
            ])
        else:
            # No graphics, just copy
            cmd.extend(["-c:v", "copy"])

        # Audio and output settings
        cmd.extend([
            "-c:a", "aac", "-b:a", "192k",
            "-b:v", "2M", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path
        ])

        logger.info(f"Graphics FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            logger.info("Graphics applied successfully with audio preservation")
            return True
        else:
            logger.error(f"FFmpeg graphics error: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg graphics process timed out")
        return False
    except Exception as e:
        logger.error(f"Error applying graphics: {str(e)}")
        return False

def test_video_streams(video_path, label):
    """Test video streams and report audio presence"""
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
            
            has_video = False
            has_audio = False
            
            for stream in streams:
                if stream.get('codec_type') == 'video':
                    has_video = True
                elif stream.get('codec_type') == 'audio':
                    has_audio = True
            
            logger.info(f"{label} video analysis - Video: {has_video}, Audio: {has_audio}")
            return has_audio
        else:
            logger.error(f"Failed to analyze {label} video")
            return False
            
    except Exception as e:
        logger.error(f"Error testing {label} video: {str(e)}")
        return False
