import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    print("🔹 Raw Columns:", df.columns.tolist())

    df.columns = ["target", "GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ"]

    print("🔹 Renamed Columns:", df.columns.tolist())
    print("\n🔹 Data Shape:", df.shape)

    return df


def clean_data(df):
    print("\n🔹 Checking missing values:\n", df.isnull().sum())
    df = df.dropna()
    print("\n🔹 Shape after cleaning:", df.shape)
    return df


def add_weather(df):
    print("\n🔹 Adding synthetic weather...")
    conditions = ["Clear", "Rain", "Fog"]
    df["weather"] = np.random.choice(conditions, size=len(df), p=[0.6, 0.3, 0.1])

    weather_map = {"Clear": 0, "Rain": 1, "Fog": 2}
    df["weather"] = df["weather"].map(weather_map)

    print("🔹 Weather sample:\n", df["weather"].head())
    return df


def normalize_data(df):
    print("\n🔹 Normalizing features...")
    scaler = StandardScaler()

    feature_cols = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ", "weather"]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    print("🔹 Normalization done")
    return df, scaler


# 🔥 CONTROLLED OVERLAP
def create_sequences_stride(df, seq_len, stride):
    X, y = [], []

    feature_cols = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ", "weather"]
    target_col = "target"

    data = df[feature_cols].values
    labels = df[target_col].values

    for i in range(0, len(df) - seq_len, stride):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len-1])

    return np.array(X), np.array(y)

def create_sequences_stride_with_groups(df, seq_len, stride):
    X, y, groups = [], [], []

    feature_cols = ["GyroX", "GyroY", "GyroZ", "AccX", "AccY", "AccZ", "weather"]
    target_col = "target"

    data = df[feature_cols].values
    labels = df[target_col].values

    for i in range(0, len(df) - seq_len, stride):
        X.append(data[i:i+seq_len])
        y.append(labels[i+seq_len-1])

        # 🔥 group id = coarse time bucket (prevents near-duplicates across splits)
        groups.append(i // seq_len)

    return np.array(X), np.array(y), np.array(groups)