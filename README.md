# A Safety-Aware and Explainable Hybrid Framework for Driving Risk Prediction using BiLSTM and Reinforcement Learning

---
## Abstract
The Real-time prediction and prevention of risky
driving behavior pose a significant challenge to intelligent trans
portation systems. Although deep learning techniques dramat
ically improve the ability to detect abnormalities, conventional
deep learning-based models tend to be black boxes that cannot
provide a transparent decision-making process nor guarantee safe
operation under failure conditions. This paper proposes a novel
architecture that provides transparency in its risk prediction
capability along with simultaneous generation of advice for
drivers on how to correct their hazardous behavior. The system
uses BiLSTM combined with a temporal attention mechanism
to process smartphone inertial sensor data stream and not only
classifies the risk state, but also precisely pinpoints the exact
moment when the risky behavior happens. Then, Q-Learning
algorithms take the predicted risky state as input and output
the necessary advice for driver corrections in a timely fashion.
Additionally, since neural networks can behave unpredictably in
very rare situations, the system employs a deterministic safety
check that takes over the probability estimation task and makes
decisions based purely on physics. Experimental results show that
the proposed system achieves 80.2% test accuracy and 0.92 AUC.

## Overview

An intelligent driving risk analysis system that combines multiple AI techniques to predict, explain, and provide corrective recommendations for driving behavior.

| Component | Technology |
|---|---|
| Risk Prediction | Deep Learning (LSTM) |
| Interpretability | Explainable AI (Attention Mechanism) |
| Action Recommendations | Reinforcement Learning (Q-Learning) |
| Reliability | Rule-Based Safety Layer |

---

## Key Features

- **Time-series risk prediction** using LSTM
- **Attention-based explainability** — understand *why* a prediction was made
- **RL-based corrective suggestions** — actionable recommendations for safer driving
- ️ **Hybrid safety override** — rule-based layer for edge cases
- ️ **Interactive Streamlit interface** with real-time input simulation
- **Attention weight visualization**

---

## ️ System Architecture

```
User Input
↓
Synthetic Time-Series Generator
↓
LSTM Model (Prediction)
↓
Attention Layer (Explanation)
↓
RL Agent (Recommendation)
↓
Rule-Based Safety Layer
↓
Final Output (UI)
```

---

## Project Structure

```
driving-risk-lstm/
│
├── src/
│ ├── model.py
│ ├── predict.py
│ ├── preprocessing.py
│ ├── explain.py
│ └── inference.py
│
├── rl/
│ ├── agent.py
│ ├── env.py
│ ├── train_rl.py
│ └── inference_rl.py
│
├── app.py
├── app2.py
├── main.py
├── test_predict.py
├── best_lstm_model.pt
├── rl_agent.pkl
├── requirements.txt
├── results/
└── report.pdf
```

---

## Dataset

- **Source:** [Smartphone Sensor-Based Driving Behavior Dataset (Kaggle)](https://www.kaggle.com/)
- **Type:** Multivariate time-series
- **Features Used:**
- Gyroscope: `GyroX`, `GyroY`, `GyroZ`
- Accelerometer: `AccX`, `AccY`, `AccZ`
- Target label (binary classification)

### Data Preprocessing

- Target column standardization
- Missing value handling
- Feature normalization
- Multi-class to binary conversion
- Sequence generation (length = 20)
- Data augmentation

---

## Models

### Model 1: LSTM (Risk Prediction)

**Architecture:**
- Input: `(20 timesteps × 7 features)`
- Stacked LSTM layers
- Attention mechanism
- Fully connected layer
- Output: Binary classification (SAFE / RISKY)

**Why LSTM?**
- Captures temporal dependencies in driving behavior
- Learns sequential patterns over time

### Explainability (Attention Mechanism)

The attention layer identifies the most critical time steps and driving patterns contributing to the prediction.

**Example attention outputs:**
- Sudden braking detected
- Sharp turning observed
- Risk concentrated at specific timesteps

---

### Model 2: Reinforcement Learning (Q-Learning)

**Purpose:** Provide corrective actions for safer driving.

**State Space:**
| Feature | Description |
|---|---|
| Acceleration | Current acceleration level |
| Braking | Braking intensity |
| Turning | Turning behavior |
| Weather | Environmental condition |

**Action Space:**
1. Maintain current driving
2. Reduce acceleration
3. Apply controlled braking
4. Smooth turning

**Reward Strategy:**
- Safe behavior → Positive reward
- Risky behavior → Negative reward

---

### Rule-Based Safety Layer

A deterministic override layer ensures reliability in edge cases:

| Condition | Classification |
|---|---|
| High acceleration + Rain | Risky |
| Sharp turning + Fog | Safety Alert |

---

## ️ Streamlit Application

### Input Parameters

- Acceleration level (Low / Medium / High)
- Braking intensity
- Turning behavior
- Weather condition

### Output

- Prediction: **SAFE** or **RISKY**
- Confidence score
- Natural language explanation
- RL-based recommendations
- Attention weight visualization

### Sample Output

```
Prediction: RISKY
Confidence: 0.94

Explanation:
- High acceleration detected
- Sharp turning observed
- Rain increases risk

Recommendations:
- Reduce acceleration
- Smooth turning
- Drive cautiously in wet conditions
```

---

## Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/driving-risk-lstm.git
cd driving-risk-lstm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

| Task | Command |
|---|---|
| Run prediction script | `python test_predict.py` |
| Run main pipeline | `python main.py` |
| Run RL agent | `python -m rl.test_agent` |
| Launch web app | `streamlit run app2.py` |

---

## Deployment

**Live Application:** [https://driving-risk-prediction.onrender.com/](https://driving-risk-prediction.onrender.com/)

**Start command:**

```bash
streamlit run app2.py --server.port $PORT --server.address 0.0.0.0
```

---

## Testing Strategy

- Basic scenario testing
- Edge case validation
- Consistency checks
- Model alignment verification
- Explanation validation

---

## Key Contributions

1. **LSTM-based temporal prediction** with attention explainability
2. **Reinforcement Learning** for intelligent decision support
3. **Hybrid system** combining prediction, explanation, and action
4. **Deployment-ready** interactive Streamlit application

---

## Future Enhancements

- [ ] Real-time sensor integration
- [ ] Continuous risk scoring system
- [ ] Deep Reinforcement Learning upgrade
- [ ] Weather API integration
- [ ] Analytics dashboard

---

## Author

**Deepank Reddy A**

---

## License

This project is for academic and research purposes.
