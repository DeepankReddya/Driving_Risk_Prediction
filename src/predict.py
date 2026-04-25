import numpy as np
import torch
import pandas as pd
from src.inference import load_model

model, scaler = load_model()

feature_cols = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ", "weather"]

def predict_sequence(seq):
    df = pd.DataFrame(seq, columns=feature_cols)

    seq_scaled = scaler.transform(df)
    seq_scaled = torch.tensor(seq_scaled.reshape(1, 20, 7), dtype=torch.float32)

    with torch.no_grad():
        output, attn = model(seq_scaled)

        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob).item()

    return pred, float(prob.max()), attn.numpy()