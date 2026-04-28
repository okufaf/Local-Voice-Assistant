import warnings
import numpy as np
import torch
import torchaudio as ta
from transformers import AutoProcessor, BarkModel

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated",
)


class TextToSpeechService:
    def __init__(
        self,
        device: str | None = None,
        checkpoint: str = "suno/bark-small",
        voice_preset: str = "v2/ru_speaker_1",
    ):
        """
        Bark TTS service with simplified sentence splitting (no NLTK).
        """

        # Select device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        self.checkpoint = checkpoint
        self.voice_preset = voice_preset

        self._patch_torch_load()

        self.processor = AutoProcessor.from_pretrained(checkpoint)
        self.model = BarkModel.from_pretrained(checkpoint).to(self.device)
        self.sample_rate = self.model.generation_config.sample_rate

    def _patch_torch_load(self):
        """Ensures Bark loads even if weights were saved on CUDA."""
        map_location = torch.device(self.device)

        if not hasattr(torch, "_original_load"):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = map_location
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

    def synthesize(self, text: str):
        """
        Synthesizes audio using Bark.
        """

        inputs = self.processor(
            text,
            voice_preset=self.voice_preset,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            wav = self.model.generate(**inputs)

        audio_array = wav.cpu().numpy().squeeze()
        return self.sample_rate, audio_array

    def long_form_synthesize(self, text: str):
        """
        Splits long text using simple punctuation (no NLTK).
        """

        if not text.strip():
            return self.sample_rate, np.zeros(int(0.5 * self.sample_rate))

        # Простое и быстрое разделение
        separators = [".", "!", "?", "\n"]
        for sep in separators:
            text = text.replace(sep, "|")

        sentences = [s.strip() for s in text.split("|") if s.strip()]

        silence = np.zeros(int(0.25 * self.sample_rate))
        pieces = []

        for sent in sentences:
            sample_rate, audio_array = self.synthesize(sent)
            pieces.append(audio_array)
            pieces.append(silence.copy())

        return self.sample_rate, np.concatenate(pieces)

    def save_voice_sample(self, text: str, output_path: str):
        """
        Saves synthesized audio to a WAV file.
        """

        sample_rate, audio_array = self.synthesize(text)
        tensor_audio = torch.tensor(audio_array).unsqueeze(0)
        ta.save(output_path, tensor_audio, sample_rate)
