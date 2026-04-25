# 🚗 Driving Risk AI System

### Explainable Driving Risk Prediction with Reinforcement Learning-Based Decision Support

---

## 📌 Overview

This project presents an **intelligent driving risk analysis system** that combines:

* 🧠 **Deep Learning (LSTM)** for risk prediction
* 🔍 **Explainable AI (Attention Mechanism)** for interpretability
* 🤖 **Reinforcement Learning (Q-Learning)** for action recommendations
* ⚠️ **Rule-based Safety Layer** for reliability

The system not only predicts whether a driving pattern is **SAFE or RISKY**, but also explains *why* and suggests *how to improve it*.

---

## 🎯 Key Features

* ✅ Time-series risk prediction using LSTM
* ✅ Attention-based explainability
* ✅ RL-based corrective suggestions
* ✅ Hybrid safety override mechanism
* ✅ Interactive Streamlit UI
* ✅ Real-time input simulation
* ✅ Visualization of attention weights

---

## 🧠 System Architecture

```text
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

## 📊 Dataset

* **Source:** Smartphone Sensor-Based Driving Behavior Dataset (Kaggle)
* **Type:** Multivariate time-series

### Features Used:

* Gyroscope: `GyroX`, `GyroY`, `GyroZ`
* Accelerometer: `AccX`, `AccY`, `AccZ`
* Target label (converted to binary)

---

## ⚙️ Data Preprocessing

* Renamed target column
* Removed missing values
* Feature normalization
* Converted multi-class labels → binary classification
* Generated time-series sequences (length = 20)
* Applied data augmentation

---

## 🤖 Model 1: LSTM (Risk Prediction)

### Architecture:

* Input: `(20 timesteps × 7 features)`
* Stacked LSTM layers
* Attention mechanism
* Fully connected layer
* Output: Binary classification

### Why LSTM?

* Captures **temporal dependencies**
* Learns **driving behavior patterns over time**

---

## 🔍 Explainability (Attention Mechanism)

The attention layer highlights:

* Important time steps
* Critical driving events

### Example Output:

* Sudden braking detected
* Sharp turning observed
* Risk concentrated at timestep 10

---

## 🤖 Model 2: Reinforcement Learning (Q-Learning)

### Purpose:

Provide **corrective actions** for safer driving

---

### Environment Design:

State includes:

* Acceleration
* Braking
* Turning
* Weather

---

### Actions:

* Maintain driving
* Reduce acceleration
* Apply controlled braking
* Smooth turning

---

### Reward Strategy:

* Safe behavior → Positive reward
* Risky behavior → Negative reward

---

## ⚠️ Rule-Based Safety Layer

To ensure reliability, a rule-based system:

* Overrides incorrect predictions
* Handles unseen edge cases

### Example:

* High acceleration + Rain → Force RISKY
* Sharp turning + Fog → Safety alert

---

## 🖥️ Streamlit Application

### Input Parameters:

* Acceleration (Low / Medium / High)
* Braking intensity
* Turning behavior
* Weather condition

---

### Output:

* Prediction (SAFE / RISKY)
* Confidence score
* Explanation (human-readable)
* RL-based recommendations
* Attention graph

---

## 📈 Sample Output

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
- Drive carefully on wet roads
```

---

## 📁 Project Structure

```
driving-risk-lstm/
│
├── src/                  # Core ML components
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── explain.py
│   └── inference.py
│
├── rl/                   # Reinforcement Learning
│   ├── agent.py
│   ├── env.py
│   ├── train_rl.py
│   └── inference_rl.py
│
├── app.py                # Stable application
├── app2.py               # Enhanced UI version
├── best_lstm_model.pt    # Trained model
├── rl_agent.pkl          # RL policy
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/driving-risk-lstm.git
cd driving-risk-lstm
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

---

## 🌐 Deployment (Render)

Use the following start command:

```bash
streamlit run app2.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🧪 Testing Strategy

The system was evaluated using:

* ✔ Basic sanity tests (safe vs risky inputs)
* ✔ Edge case testing
* ✔ Consistency checks
* ✔ LSTM vs RL alignment
* ✔ Explanation validation

---

## 🏆 Key Contributions

This project integrates:

* Deep Learning (LSTM)
* Reinforcement Learning (Q-Learning)
* Explainable AI (Attention)
* Rule-based Safety System

👉 Result: **End-to-end intelligent driving assistant system**

---

## 🚀 Future Enhancements

* Real-time sensor data integration
* Risk scoring (0–100 scale)
* Deep RL (DQN)
* Live weather API
* Cloud deployment with analytics

---

## 🎤 Viva Explanation

> “This system combines temporal deep learning for risk prediction with reinforcement learning for corrective decision-making, enhanced with explainability and safety mechanisms.”

---

## 👨‍💻 Author

**Deepank Reddy A**

---

## ⭐ Project Status

```text
Production-ready | Research-level | Fully functional
```
