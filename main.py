import argparse
import os
import threading
import time
from queue import Queue
from dotenv import load_dotenv

import numpy as np
import transformers
import sounddevice as sd
import whisper
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from rich.console import Console

from sentiment import load_model, SentimentPrediction
from tts_service import TextToSpeechService

transformers.logging.set_verbosity_error()
load_dotenv()

project_root = os.getenv("PROJECT_ROOT")

os.environ["HF_HOME"] = os.path.join(project_root, os.getenv("HF_HOME"))
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["XDG_CACHE_HOME"] = os.path.join(project_root, os.getenv("XDG_CACHE_HOME"))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(project_root, os.getenv("TRANSFORMERS_CACHE"))
os.environ["TORCH_HOME"] = os.path.join(project_root, os.getenv("TORCH_HOME"))

console = Console()

stt = whisper.load_model("medium")
tts = TextToSpeechService()
sentiment_model = load_model()

parser = argparse.ArgumentParser(description="Local Voice Assistant with Bark TTS")
parser.add_argument("--model", type=str, default="stepfun/step-3.5-flash:free")
parser.add_argument("--save-voice", action="store_true")
args = parser.parse_args()


prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and friendly AI assistant. 
    You are polite, respectful, and aim to provide concise responses of less than 20 words."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(
    model=args.model,
    temperature=0.6,
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=15
)

chain = prompt | llm

chat_sessions = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Get or create chat history for a session.

    Args:
        session_id (str): Unique session identifier.

    Returns:
        InMemoryChatMessageHistory: The chat history object.
    """
    if session_id not in chat_sessions:
        chat_sessions[session_id] = InMemoryChatMessageHistory()
    return chat_sessions[session_id]


chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)


def record_audio(stop_event, data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata, frames, time, status):
        if status:
            console.print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=16000, dtype="int16", channels=1, callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)


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


def get_llm_response(text: str) -> str:
    """
    Generates a response to the given text using the language model.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The generated response.
    """
    session_id = "voice_assistant_session"
    response = chain_with_history.invoke(
        {"input": text},
        config={"session_id": session_id}
    )
    return (response.content or "").strip()


def play_audio(sample_rate, audio_array):
    """
    Plays the given audio data using the sounddevice library.

    Args:
        sample_rate (int): The sample rate of the audio data.
        audio_array (numpy.ndarray): The audio data to be played.

    Returns:
        None
    """
    sd.play(audio_array, sample_rate)
    sd.wait()


def analyze_emotion(text: str) -> SentimentPrediction:
    """
    Emotion analysis for dynamically adjusting exaggeration.

    Returns a SentimentPrediction object containing:
        sentiment_label: the mood label (e.g., 'positive', 'negative', 'neutral')
        sentiment_score: a numeric mood score from the model
    """

    if not text:
        return SentimentPrediction("neutral", 0.0)

    text = text.strip()
    sentiment = sentiment_model(text)
    return SentimentPrediction(
        label=sentiment.label,
        score=sentiment.score
    )


if __name__ == "__main__":
    console.print("[cyan]🤖 Local Voice Assistant with Bark TTS")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0

    try:
        while True:
            console.input("🎤 Press Enter to start recording, then press Enter again to stop.")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue)
            )
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_array = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_array.size == 0:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")
                continue

            with console.status("Transcribing...", spinner="dots"):
                text = transcribe(audio_array)

            console.print(f"[yellow]You: {text}")

            with console.status("Generating response...", spinner="dots"):
                response = get_llm_response(text)

                sentiment = analyze_emotion(response)
                label = sentiment.label
                score = sentiment.score

                sample_rate, audio_out = tts.long_form_synthesize(response)

            console.print(f"[cyan]Assistant: {response}")
            console.print(f"[dim](Emotion: {label}, Score: {score:.2f})[/dim]")

            if args.save_voice:
                response_count += 1
                filename = f"voices/response_{response_count:03d}.wav"
                tts.save_voice_sample(response, filename)
                console.print(f"[dim]Voice saved to: {filename}[/dim]")

            play_audio(sample_rate, audio_out)

    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended. Thank you for using the Voice Assistant!")
