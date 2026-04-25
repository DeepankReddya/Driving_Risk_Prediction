from src.preprocessing import (
    load_data, clean_data, add_weather, normalize_data,
    create_sequences_stride_with_groups
)

from src.model import LSTMModel

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

# -----------------------------
# LOAD + CLEAN
# -----------------------------
df = load_data("data/sensor_raw.csv")
df = clean_data(df)

print("\n🔴 Original Target:", df["target"].unique())

df["target"] = df["target"].apply(lambda x: 0 if x <= 2 else 1)

print("🟢 Fixed Target:", df["target"].unique())

# -----------------------------
# ADD WEATHER + NORMALIZE
# -----------------------------
df = add_weather(df)
df, scaler = normalize_data(df)

# -----------------------------
# SEQUENCES + GROUPS
# -----------------------------
SEQ_LEN = 20
STRIDE = 5

X, y, groups = create_sequences_stride_with_groups(df, SEQ_LEN, STRIDE)

print("\n🔹 Sequence shape:", X.shape)

# -----------------------------
# GROUP SPLIT
# -----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print("\n🔹 Train:", X_train.shape)
print("🔹 Test:", X_test.shape)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# -----------------------------
# MODEL
# -----------------------------
model = LSTMModel()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# TRAINING
# -----------------------------
EPOCHS = 40
BATCH_SIZE = 32

best_acc = 0

for epoch in range(EPOCHS):

    # Shuffle
    perm = torch.randperm(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    model.train()

    for i in range(0, len(X_train), BATCH_SIZE):
        xb = X_train[i:i+BATCH_SIZE]
        yb = y_train[i:i+BATCH_SIZE]

        outputs, _ = model(xb)   # 🔥 FIX HERE
        loss = loss_fn(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X_test)   # 🔥 FIX HERE
        preds = torch.argmax(outputs, dim=1)

        acc = accuracy_score(y_test.numpy(), preds.numpy())

    print(f"Epoch {epoch+1}/{EPOCHS} | Accuracy: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_lstm_model.pt")

print(f"\n✅ Best Accuracy: {best_acc:.4f}")


# -----------------------------
# TEST ATTENTION
# -----------------------------
model.eval()

sample = X_test[0].unsqueeze(0)

with torch.no_grad():
    output, attn = model(sample)

    prob = torch.softmax(output, dim=1)
    pred = torch.argmax(prob).item()

print("\n🔍 SAMPLE PREDICTION")
print("Prediction:", "RISKY" if pred == 1 else "SAFE")
print("Confidence:", prob.max().item())

attn = attn.squeeze().numpy()

print("\n📊 Attention Weights:")
print(attn)

# -----------------------------
# ATTENTION VISUALIZATION
# -----------------------------
import matplotlib.pyplot as plt

attn = attn.squeeze()  # (20,)

plt.figure(figsize=(10, 4))
plt.plot(attn, marker='o')
plt.title("Attention Weights Over Time Steps")
plt.xlabel("Time Step")
plt.ylabel("Importance")
plt.grid()

plt.savefig("attention_plot.png")  # saves for report
plt.show()

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test.numpy(), preds.numpy())
print("\nConfusion Matrix:\n", cm)

from sklearn.metrics import classification_report

print("\nClassification Report:\n")
print(classification_report(y_test.numpy(), preds.numpy()))

torch.save({
    "model_state": model.state_dict(),
    "scaler": scaler
}, "best_model.pth")