import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def make_script(news_text, topic):
    """Generate news script using Gemini AI"""
    try:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            return generate_simple_script(news_text, topic)
        
        genai.configure(api_key=api_key)
        
        # Use the correct model name for current API
        model = genai.GenerativeModel('gemini-pro')  # Changed from gemini-1.5-flash
        
        prompt = f"""
        You are a professional female TV news anchor. Create a compelling 60-90 second news script based on this real news about "{topic}":

        {news_text}

        Requirements:
        - Professional, authoritative broadcast tone appropriate for a female anchor
        - Clear, engaging opening hook
        - Present key facts in logical order
        - Use broadcast-appropriate language
        - Include smooth transitions between points
        - End with a professional closing
        - Length: 150-250 words for 60-90 seconds
        - Write as direct speech to viewers
        - No special formatting, just natural speech
        - Focus on the most important developments
        - Sound conversational yet professional
        
        Create the news script now:
        """
        
        response = model.generate_content(prompt)
        script = response.text.strip()
        
        # Clean up the script
        script = clean_script(script)
        
        logger.info(f"Generated script length: {len(script)} characters")
        return script
        
    except Exception as e:
        logger.error(f"Error generating script with Gemini: {str(e)}")
        return generate_simple_script(news_text, topic)

def clean_script(script):
    """Clean and format script for TTS"""
    # Remove markdown formatting
    script = script.replace('**', '')
    script = script.replace('*', '')
    script = script.replace('#', '')
    script = script.replace('[', '')
    script = script.replace(']', '')
    
    # Fix common issues
    script = script.replace('..', '.')
    script = script.replace('? ', '? ')
    script = script.replace('! ', '! ')
    script = script.replace('…', '...')
    
    # Remove extra whitespace
    script = ' '.join(script.split())
    
    return script
def generate_simple_script(news_text, topic):
    """Generate professional script when Gemini fails"""
    # Extract key information from news text
    lines = news_text.split('\n')
    key_points = []
    
    for line in lines[:15]:  # Take first 15 lines
        if line.strip() and len(line.strip()) > 30:
            # Clean the line
            clean_line = line.strip()
            if clean_line.startswith('BREAKING:') or clean_line.startswith('UPDATE:'):
                clean_line = clean_line.split(':', 1)[1].strip()
            key_points.append(clean_line)
    
    # Use the first few key points
    main_content = ' '.join(key_points[:3]) if key_points else f"Recent developments in {topic} continue to unfold with significant implications for the industry."
    
    # Ensure we have good content
    if len(main_content) < 100:
        main_content = f"Breaking news in the {topic} sector reveals important developments that are shaping the current landscape. Industry experts are analyzing these changes as they impact various stakeholders and market conditions."
    
    script = f"""
    Hello everyone! I'm Pannu, your AI news reporter, and I'm here with the latest updates on {topic}.
    
    {main_content[:300]}
    
    These developments are really exciting and show how much is happening in the {topic} world right now! Industry experts are keeping a close eye on these changes as they could affect many different areas.
    
    It's really interesting to see how quickly things are moving, and I think we can expect even more developments in the coming days. Everyone involved is working hard to understand what this all means.
    
    I'll keep watching this story closely and bring you all the latest updates as they happen! Thanks so much for staying informed with me.
    
    This is Pannu with your AI news update. See you next time!
    """
    
    return clean_script(script)
