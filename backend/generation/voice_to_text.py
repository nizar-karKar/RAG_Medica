import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()


class AudioRecorder:
    """Records audio from the local microphone and saves it as a WAV file."""

    def __init__(self, sample_rate: int = 44100, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels

    def record(self, duration: int = 10, output_filename: Optional[str] = None) -> str:
        try:
            import sounddevice as sd
            from scipy.io.wavfile import write
        except ImportError as exc:
            raise ImportError(
                "Please install the required packages to use the microphone: "
                "pip install sounddevice scipy"
            ) from exc

        print(f"Recording audio for {duration} seconds...")
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
        )
        sd.wait()
        print("Finished recording.")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        voice_queries_dir = os.path.join(base_dir, "voice_queries")
        os.makedirs(voice_queries_dir, exist_ok=True)

        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"recording_{timestamp}.wav"
        if not output_filename.endswith(".wav"):
            output_filename += ".wav"

        output_path = os.path.join(voice_queries_dir, output_filename)
        write(output_path, self.sample_rate, recording)
        return output_path


class VoiceTranscriber:
    """Transcribes audio files via the ElevenLabs Speech-to-Text API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "scribe_v2",
        recorder: Optional[AudioRecorder] = None,
    ):
        self.api_key = api_key or os.getenv("ELEVENT_LAB_API")
        self.model_id = model_id
        self._client = ElevenLabs(api_key=self.api_key)
        self.recorder = recorder or AudioRecorder()

    def transcribe(
        self,
        audio_path: Optional[str] = None,
        language_code: Optional[str] = None,
        duration: int = 5,
    ) -> str:
        is_temp_file = False
        if audio_path is None:
            audio_path = self.recorder.record(duration=duration)
            is_temp_file = True

        try:
            with open(audio_path, "rb") as audio_file:
                response = self._client.speech_to_text.convert(
                    file=audio_file,
                    model_id=self.model_id,
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
            if is_temp_file and os.path.exists(audio_path):
                os.remove(audio_path)


def record_audio_from_mic(duration: int = 10, output_filename: Optional[str] = None) -> str:
    """Backward-compatible wrapper."""
    return AudioRecorder().record(duration=duration, output_filename=output_filename)


def transcribe_audio(
    audio_path: Optional[str] = None,
    language_code: Optional[str] = None,
    duration: int = 5,
) -> str:
    """Backward-compatible wrapper."""
    return VoiceTranscriber().transcribe(
        audio_path=audio_path, language_code=language_code, duration=duration
    )
