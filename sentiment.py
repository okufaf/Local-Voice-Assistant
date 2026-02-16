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

