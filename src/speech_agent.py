import speech_recognition as sr
import io

class VoiceAgent:
    """
    Voice Agent for the Shopping Assistant.
    
    This module uses the Google Speech Recognition API (via the SpeechRecognition library)
    to convert spoken words from an audio file into text. It receives the audio file 
    recorded by the user's browser via the Flask backend.
    """
    
    def __init__(self):
        # Initialize the recognizer
        self.recognizer = sr.Recognizer()

    def transcribe_audio(self, audio_file_bytes):
        """
        Transcribes audio bytes (e.g., WAV format) using Google Speech Recognition.
        
        Args:
            audio_file_bytes (bytes): The raw audio file bytes received from the frontend.
            
        Returns:
            dict: A status dictionary containing 'success', 'text', and 'error'.
        """
        # Dictionary to return results
        response = {
            "success": True,
            "text": None,
            "error": None
        }
        
        try:
            print("Processing audio file bytes...")
            # Convert bytes to a file-like object that SpeechRecognition can read
            audio_file = io.BytesIO(audio_file_bytes)
            
            with sr.AudioFile(audio_file) as source:
                # Capture the audio data from the file
                audio_data = self.recognizer.record(source)
                
        except Exception as e:
            response["success"] = False
            response["error"] = f"Failed to process audio file: {str(e)}"
            return response

        # Send the audio data to Google's Speech Recognition API
        try:
            print("Transcribing via Google Speech Recognition API...")
            text = self.recognizer.recognize_google(audio_data)
            response["text"] = text
            print(f"Transcription successful: '{text}'")
            
        except sr.RequestError as e:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = f"Google Speech Recognition API unavailable: {e}"
            
        except sr.UnknownValueError:
            # Speech was unintelligible
            response["success"] = False
            response["error"] = "Unable to recognize speech. Please try speaking clearer."

        return response
