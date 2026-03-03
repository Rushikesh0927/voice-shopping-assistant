import os
import sys
from flask import Flask, render_template, request, jsonify

# Add src to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.nlp_engine import ProductRecommender
from src.speech_agent import VoiceAgent

app = Flask(__name__)

# Use absolute paths so it works no matter where the script is run from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "Amazon-Products.csv")

# Initialize the NLP recommender once on startup
print("Initializing Product Recommender...")
try:
    # Use 500,000 rows to ensure a vast catalog of products for accurate matching
    recommender = ProductRecommender(data_path=DATA_PATH, max_rows=500000)
    recommender_status = "Ready"
except Exception as e:
    print(f"Error initializing recommender: {e}")
    recommender = None
    recommender_status = "Error"

# Initialize Voice Agent
voice_agent = VoiceAgent()

@app.route('/')
def home():
    """Renders the main dashboard interface."""
    return render_template('index.html', model_status=recommender_status)

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    """
    Receives an audio file (WAV format) from the frontend, uses Google Speech Recognition
    API to transcribe it, and returns the top product recommendations.
    """
    if 'audio' not in request.files:
        return jsonify({"success": False, "error": "No audio file provided"}), 400
        
    audio_file = request.files['audio']
    audio_bytes = audio_file.read()
    
    print("Received audio file from frontend. Size:", len(audio_bytes), "bytes")
    
    # Send to Voice Agent for transcription
    transcription_result = voice_agent.transcribe_audio(audio_bytes)
    
    if not transcription_result['success']:
        print("Transcription Failed:", transcription_result['error'])
        # If the API fails, return the error to the UI
        return jsonify({
            "success": False, 
            "error": transcription_result['error'],
            "text": None,
            "recommendations": []
        })
        
    transcribed_text = transcription_result['text']
    print(f"Transcription Result: '{transcribed_text}'")
    
    # Process the transcribed text using the NLP engine
    if recommender:
        recommendations = recommender.recommend_products(transcribed_text, top_k=6)
        return jsonify({
            "success": True,
            "text": transcribed_text,
            "recommendations": recommendations,
            "error": None
        })
    else:
        return jsonify({
            "success": False,
            "error": "NLP Engine is not initialized. Check server logs.",
            "text": transcribed_text,
            "recommendations": []
        }), 500

@app.route('/api/search_text', methods=['POST'])
def search_text():
    """
    Fallback endpoint to allow text-based searches through the UI.
    """
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({"success": False, "error": "Query is empty"}), 400
        
    if recommender:
        recommendations = recommender.recommend_products(query, top_k=6)
        return jsonify({
            "success": True,
            "recommendations": recommendations,
            "text": query
        })
    else:
        return jsonify({"success": False, "error": "NLP Engine is down."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
