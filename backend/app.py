from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import logging
from datetime import datetime
from dotenv import load_dotenv
from news_fetch import get_news
from script_gen import make_script
from tts_gen import make_speech
from lip_sync import make_video
from video_make import add_graphics

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Ensure directories exist
os.makedirs('temp', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/generate', methods=['POST'])
def generate_video():
    try:
        data = request.json
        if not data or 'topic' not in data:
            return jsonify({"error": "Topic is required"}), 400

        topic = data.get('topic', '').strip()
        if not topic:
            return jsonify({"error": "Topic cannot be empty"}), 400

        session_id = str(uuid.uuid4())[:8]
        logger.info(f"Starting video generation for: {topic} (Session: {session_id})")

        # Step 1: Fetch real news
        logger.info("Step 1: Fetching news...")
        news_data = get_news(topic)
        if not news_data:
            return jsonify({"error": "Could not fetch news for this topic"}), 404

        # Step 2: Generate script
        logger.info("Step 2: Generating script...")
        script = make_script(news_data, topic)
        if not script:
            return jsonify({"error": "Failed to generate script"}), 500

        # Step 3: Generate text-to-speech
        logger.info("Step 3: Generating speech...")
        audio_file = make_speech(script, session_id)
        if not audio_file:
            return jsonify({"error": "Failed to generate speech"}), 500

        # Step 4: Generate lip-synced video
        logger.info("Step 4: Creating lip-synced video...")
        video_file = make_video(audio_file, session_id)
        if not video_file:
            return jsonify({"error": "Failed to create video"}), 500

        # Step 5: Add graphics and finalize
        logger.info("Step 5: Adding graphics...")
        final_video = add_graphics(video_file, topic, session_id)
        if not final_video:
            return jsonify({"error": "Failed to add graphics"}), 500

        logger.info(f"Video generation completed: {final_video}")

        return send_file(
            final_video,
            as_attachment=True,
            download_name=f"news_report_{session_id}.mp4",
            mimetype='video/mp4'
        )

    except Exception as e:
        logger.error(f"Error in video generation: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting News Reporter AI backend on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)
