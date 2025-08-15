# A Hybrid Deep Learning Framework for Epidemiological Forecasting & Optimal Control

[cite_start]This repository contains the implementation of the paper, "A Hybrid Deep Learning Framework for Spatiotemporal Epidemiological Forecasting and Optimal Control"[cite: 1]. The project develops a sophisticated, data-driven system to both predict the spread of epidemics and learn optimal intervention strategies.

## 📋 Project Overview

[cite_start]Traditional epidemiological models like SEIR are often limited by static parameters and an inability to capture complex, real-world dynamics[cite: 10, 11]. This project addresses these limitations by implementing a novel, two-part deep learning framework:

1.  **A Predictive Engine**: A **Graph Neural Network (GNN)** is combined with a **Long Short-Term Memory (LSTM)** network to produce high-fidelity, spatiotemporal forecasts of disease progression. [cite_start]The GNN models the connections between geographical regions, while the LSTM captures temporal patterns[cite: 14].
2.  **A Prescriptive Agent**: A **Deep Reinforcement Learning (RL)** agent uses the trained GNN-LSTM as a simulated environment. [cite_start]It learns to recommend optimal non-pharmaceutical interventions (NPIs) that balance minimizing disease spread with socioeconomic costs[cite: 16].

[cite_start]By unifying predictive and prescriptive analytics, this framework provides a powerful tool for strategic public health planning[cite: 19].

---

## 📊 Key Results

### Predictive Model Performance
The GNN-LSTM model was trained on state-wise COVID-19 data for India. After addressing challenges like overfitting through architectural improvements and log-transformation of data, the model achieved a strong predictive performance on the unseen test set.

* **Test Set Mean Absolute Error (MAE)**: 56,962.77 cases
* **Test Set Root Mean Squared Error (RMSE)**: 111,137.56 cases

![Delhi Forecast](/assets/lstm-result.png)
*Figure 1: A 7-day forecast for Delhi. The model (red) successfully captures the upward trend of the actual case data (blue), demonstrating its ability to learn temporal dynamics.*

### Optimal Control Policy
The DQN agent was trained for 200 episodes, learning to make intervention decisions over a 180-day simulated period. The agent's performance improved significantly, indicating it learned an effective policy.

![RL Reward Curve](/assets/RL-result.png)
*Figure 2: The agent's cumulative reward per episode shows a clear upward trend, demonstrating successful learning of a policy that improves public health outcomes.*

---

## 📂 Repository Structure

```
├── nbks/
│   ├── GNN_LSTM_MODEL.ipynb            # Notebook for training the predictive model
│   ├── Deep_Reinforcement_Learning...ipynb # Notebook for training the RL agent
│   └── basemodel.ipynb                 # Notebook for the baseline agent-based model
├── models/
│   ├── gnn_lstm_model.pth              # Saved weights for the trained GNN-LSTM
│   ├── rl_agent_q_network.pth          # Saved weights for the trained RL agent
│   ├── data_scaler.gz                  # Saved scaler for data normalization
│   └── project_params.pth              # Other necessary parameters
├── DL_epidemology v4.pdf                 # The source research paper
└── README.md                           # This file
```

---

## 🚀 How to Run

### 1. Prerequisites
- Python 3.8+
- A Google Colab account (for GPU acceleration)

### 2. Setup
Clone the repository to your local machine or Google Drive:
```bash
git clone [https://github.com/katakuri004/epidemology-project.git](https://github.com/katakuri004/epidemology-project.git)
cd epidemology-project
```

### 3. Install Dependencies
Create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:
```
pandas
numpy
scikit-learn
torch
torch_geometric
matplotlib
joblib
```

### 4. Running the Notebooks
The project is split into two main notebooks that should be run in order.

**Part 1: Train the Predictive Model**
- Open `nbks/GNN_LSTM_MODEL.ipynb`.
- This notebook will process the raw epidemiological data, define and train the `SpatioTemporalGNN_v2` model, and save the final trained model, scaler, and parameters to the `/models` directory.

**Part 2: Train the RL Agent**
- Open `nbks/Deep_Reinforcement_Learning_(RL)_agent_for_optimal_control.ipynb`.
- This notebook will load the trained artifacts from Part 1.
- It then defines the `EpidemicEnv` and `DQNAgent` and runs the RL training loop to discover an optimal intervention policy.