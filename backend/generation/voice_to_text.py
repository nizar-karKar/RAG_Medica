import os
from datetime import datetime
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()


def record_audio_from_mic(duration: int = 10, output_filename: str = None) -> str:
    try:
        import sounddevice as sd
        from scipy.io.wavfile import write
    except ImportError:
        raise ImportError("Please install the required packages to use the microphone: pip install sounddevice scipy")

    fs = 44100  # Sample rate
    print(f"Recording audio for {duration} seconds...")
    
    # Record audio (channels=1 for mono)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Finished recording.")

    # Determine the output path
    # __file__ is in backend/generation/, so its grandparent is backend/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    voice_queries_dir = os.path.join(base_dir, "voice_queries")
    os.makedirs(voice_queries_dir, exist_ok=True)

    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"recording_{timestamp}.wav"
    
    if not output_filename.endswith(".wav"):
        output_filename += ".wav"

    output_path = os.path.join(voice_queries_dir, output_filename)
    
    # Save as WAV file
    write(output_path, fs, recording)
    
    return output_path

def transcribe_audio(audio_path: str = None, language_code: str = None, duration: int = 5) -> str:
    is_temp_file = False
    if audio_path is None:
        audio_path = record_audio_from_mic(duration=duration)
        is_temp_file = True
    
    client = ElevenLabs(api_key=os.getenv("ELEVENT_LAB_API"))

    try:
        with open(audio_path, "rb") as audio_file:
            response = client.speech_to_text.convert(
                file=audio_file,
                model_id="scribe_v2",               
                language_code=language_code,         
                diarize=False,                       
                timestamps_granularity="word",       
                tag_audio_events=False,              
            )
            
        transcript = response.text
        print("\n--- Transcript ---")
        print(transcript)
        print("------------------\n")

        return transcript
    finally:
        # Clean up the audio file if we recorded it
        if is_temp_file and os.path.exists(audio_path):
            os.remove(audio_path)
                
# if __name__ == "__main__":
#     print("Testing complete voice-to-text pipeline...")
#     # This will automatically record for 5 seconds and then transcribe and print the result.
#     transcribed_audio=transcribe_audio()
#     print(transcribed_audio)