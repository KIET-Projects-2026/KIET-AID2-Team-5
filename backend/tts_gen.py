from TTS.api import TTS
import os
import logging
import soundfile as sf
import numpy as np

logger = logging.getLogger(__name__)

# Initialize TTS model globally
tts_model = None

def initialize_tts():
    """Initialize TTS model"""
    global tts_model
    try:
        # Try primary model first
        tts_model = TTS("tts_models/en/ljspeech/glow-tts")
        logger.info("TTS model initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing primary TTS model: {str(e)}")
        try:
            # Try backup model
            tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("Backup TTS model initialized")
            return True
        except Exception as e2:
            logger.error(f"Failed to initialize any TTS model: {str(e2)}")
            return False

def make_speech(script, session_id):
    """Generate speech from script using Coqui TTS"""
    global tts_model
    
    try:
        # Initialize TTS if not done
        if tts_model is None:
            if not initialize_tts():
                logger.error("No TTS model available")
                return None
        
        # Clean script for TTS
        cleaned_script = clean_text_for_tts(script)
        
        # Generate output path
        output_path = f"temp/{session_id}_speech.wav"
        
        # Generate speech
        tts_model.tts_to_file(text=cleaned_script, file_path=output_path)
        
        # Post-process audio
        processed_path = post_process_audio(output_path, session_id)
        
        logger.info(f"Speech generated successfully: {processed_path}")
        return processed_path
        
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return None

def clean_text_for_tts(text):
    """Clean text for better TTS pronunciation"""
    # Expand common abbreviations
    replacements = {
        'AI': 'Artificial Intelligence',
        'US': 'United States',
        'UK': 'United Kingdom',
        'CEO': 'Chief Executive Officer',
        'GDP': 'Gross Domestic Product',
        'COVID-19': 'COVID nineteen',
        'Dr.': 'Doctor',
        'Mr.': 'Mister',
        'Mrs.': 'Misses',
        'Ms.': 'Miss',
        'Inc.': 'Incorporated',
        'Corp.': 'Corporation',
        'Ltd.': 'Limited'
    }
    
    for abbr, full in replacements.items():
        text = text.replace(f' {abbr} ', f' {full} ')
        text = text.replace(f' {abbr}.', f' {full}.')
        text = text.replace(f'{abbr} ', f'{full} ')
    
    # Remove extra whitespace and clean up
    text = ' '.join(text.split())
    
    return text

def post_process_audio(input_path, session_id):
    """Post-process generated audio"""
    try:
        # Load audio
        audio, sr = sf.read(input_path)
        
        # Ensure we have valid audio
        if len(audio) == 0:
            logger.error("Generated audio is empty")
            return None
        
        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Add fade in/out
        fade_samples = int(0.05 * sr)  # 50ms fade
        if len(audio) > fade_samples * 2:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out
        
        # Save processed audio
        output_path = f"temp/{session_id}_speech_final.wav"
        sf.write(output_path, audio, sr, format='WAV', subtype='PCM_16')
        
        # Remove original
        if os.path.exists(input_path) and input_path != output_path:
            os.remove(input_path)
        
        # Verify the output file
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            logger.info(f"Audio processed successfully: {output_path} ({os.path.getsize(output_path)} bytes)")
            return output_path
        else:
            logger.error("Processed audio file is invalid or too small")
            return None
        
    except Exception as e:
        logger.error(f"Error post-processing audio: {str(e)}")
        return input_path if os.path.exists(input_path) else None
