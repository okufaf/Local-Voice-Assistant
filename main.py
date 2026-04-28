import argparse
import os
import threading
import time
from queue import Queue
from datetime import datetime

import numpy as np
import transformers

transformers.logging.set_verbosity_error()

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.rule import Rule
from rich.box import ROUNDED
from pathlib import Path

from src.agent import get_llm_response
from src.sentiment import analyze_emotion
from src.stt_service import transcribe
from src.tts_service import  TextToSpeechService
from src.utils import play_audio, record_audio

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "hf_cache"
CACHE_DIR.mkdir(exist_ok=True)

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR / "whisper")
os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")

console = Console(width=100)

tts = TextToSpeechService()

parser = argparse.ArgumentParser(description="Local Voice Assistant with Bark TTS")
parser.add_argument("--model", type=str, default="minimax/minimax-m2.5:free")
parser.add_argument("--save-voice", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    header = Panel(
        Text("🤖 LOCAL VOICE ASSISTANT", style="bold cyan", justify="center")
        + Text(f"\n⚡ Powered by Bark TTS & OpenRouter | 🕒 {datetime.now().strftime('%H:%M:%S')}",
               style="dim blue", justify="center"),
        box=ROUNDED,
        border_style="cyan",
        title="[bold magenta]VOICE AI[/bold magenta]",
        subtitle="[dim]Нажмите Ctrl+C для выхода[/dim]"
    )
    console.print(header)
    console.print(Rule(style="dim cyan"))

    if args.save_voice:
        os.makedirs("voices", exist_ok=True)

    response_count = 0
    start_time = time.time()

    try:
        while True:
            # 🎤 Запрос на запись
            console.print(Text("🎤 Готов к записи. Нажмите Enter чтобы начать, и Enter снова чтобы остановить.", style="bold yellow"))
            console.input("[bold green]▶ [/bold green]")

            data_queue = Queue()
            stop_event = threading.Event()
            recording_thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            recording_thread.start()

            input()
            stop_event.set()
            recording_thread.join()

            audio_data = b"".join(list(data_queue.queue))
            audio_array = (
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if audio_array.size == 0:
                console.print(Panel(
                    "⚠️ Аудио не записано. Проверьте микрофон и разрешения приложения.",
                    title="[bold red]Ошибка записи[/bold red]",
                    border_style="red",
                    box=ROUNDED
                ))
                continue

            with console.status("🎧 Распознавание речи...", spinner="dots", spinner_style="cyan"):
                text = transcribe(audio_array)

            console.print(Panel(
                f"🗣️ {text}",
                title="[bold yellow]ВЫ[/bold yellow]",
                border_style="yellow",
                box=ROUNDED
            ))

            with console.status("🧠 Генерация ответа...", spinner="dots", spinner_style="green"):
                response = get_llm_response(text)
                sentiment = analyze_emotion(response)
                label = sentiment.label
                score = sentiment.score
                sample_rate, audio_out = tts.long_form_synthesize(response)

            console.print(Panel(
                f"💬 {response}\n\n[dim]🎭 Эмоция: {label} | Уверенность: {score:.2f}[/dim]",
                title="[bold cyan]АССИСТЕНТ[/bold cyan]",
                border_style="cyan",
                box=ROUNDED
            ))

            if args.save_voice:
                response_count += 1
                filename = f"voices/response_{response_count:03d}.wav"
                tts.save_voice_sample(response, filename)
                console.print(Panel(
                    f"💾 Голосовой образец сохранён: {filename}",
                    title="[bold green]Сохранено[/bold green]",
                    border_style="green",
                    box=ROUNDED,
                    style="dim"
                ))

            play_audio(sample_rate, audio_out)

            elapsed = time.time() - start_time
            mins, secs = divmod(int(elapsed), 60)
            console.print(Rule(style="dim"))
            console.print(f"[dim]📊 Статистика: {response_count} ответов | ⏱️ {mins:02d}:{secs:02d}[/dim]")
            console.print(Rule(style="dim"))

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        console.print("\n")
        console.print(Panel(
            f"👋 Сессия завершена.\n⏱️ Общее время: {mins:02d}:{secs:02d}\n💬 Всего ответов: {response_count}",
            title="[bold blue]Завершение[/bold blue]",
            border_style="blue",
            box=ROUNDED
        ))

    console.print("[blue]✨ Спасибо за использование голосового ассистента!")


