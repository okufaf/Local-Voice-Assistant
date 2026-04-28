import numpy as np
import whisper

stt = whisper.load_model("medium")


def transcribe(audio_array: np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_array (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_array, fp16=False)
    return result["text"].strip()
