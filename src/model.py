import torch
import torch.nn as nn


# -----------------------------
# ATTENTION MODULE
# -----------------------------
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn(lstm_output))
        attention = self.v(energy).squeeze(-1)

        weights = torch.softmax(attention, dim=1)

        context = torch.sum(lstm_output * weights.unsqueeze(-1), dim=1)

        return context, weights


# -----------------------------
# MAIN MODEL
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=7):
        super(LSTMModel, self).__init__()

        self.hidden_size = 128

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = Attention(self.hidden_size)

        self.fc1 = nn.Linear(self.hidden_size * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        context, attn_weights = self.attention(lstm_out)

        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out, attn_weights