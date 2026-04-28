from pathlib import Path

import torch
import yaml
from pydantic import BaseModel
from transformers import pipeline

config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class SentimentPrediction(BaseModel):
    label: str
    score: float


def load_model():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_hf = pipeline(config['task'], model=config['model'], device=device)

    def model(text: str) -> SentimentPrediction:
        pred = model_hf(text)
        pred_best_class = pred[0]
        return SentimentPrediction(
            label=pred_best_class['label'],
            score=pred_best_class['score']
        )

    return model


sentiment_model = load_model()


def analyze_emotion(text: str) -> SentimentPrediction:
    """
    Emotion analysis for dynamically adjusting exaggeration.

    Returns a SentimentPrediction object containing:
        sentiment_label: the mood label (e.g., 'positive', 'negative', 'neutral')
        sentiment_score: a numeric mood score from the model
    """

    if not text:
        return SentimentPrediction("neutral", 0.5)

    text = text.strip()
    sentiment = sentiment_model(text)
    return SentimentPrediction(
        label=sentiment.label,
        score=sentiment.score
    )
