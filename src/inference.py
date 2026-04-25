import torch
from src.model import LSTMModel
from config import MODEL_PATH

def load_model():
    # 🔥 IMPORTANT FIX
    checkpoint = torch.load(MODEL_PATH, weights_only=False)

    model = LSTMModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    scaler = checkpoint["scaler"]

    return model, scaler