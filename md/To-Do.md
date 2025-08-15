# Project Roadmap & To-Do List

This document outlines the immediate action items for improving the current project and the long-term future scope for extending this research.

---

## ✅ To-Do List: Enhancing the Current Project

This is a checklist of tasks to improve the quality, reproducibility, and accuracy of the existing implementation.

### Repository and Presentation
- [ ] **Create a Comprehensive `README.md`**:
    - [ ] Add a project overview explaining the GNN-LSTM and RL components.
    - [ ] Include a "Results" section with key metrics (MAE/RMSE) and plots (forecasts, RL reward curve).
    - [ ] Write a "How to Run" section with clear, step-by-step instructions.
- [ ] **Generate a `requirements.txt` File**:
    - [ ] List all necessary Python libraries (`pandas`, `torch`, `scikit-learn`, etc.) to allow for easy environment setup using `pip install -r requirements.txt`.
- [ ] **Restructure the Code**:
    - [ ] Split the main notebook into two logical parts:
        1. `1_GNN_LSTM_Forecasting.ipynb` (Data Prep, Training, Evaluation of the predictive model).
        2. `2_RL_Optimal_Control.ipynb` (Loading the trained model, training the RL agent).

### Model Accuracy and Performance
- [ ] **Incorporate Additional Features**:
    - [ ] Modify the data preprocessing pipeline to create a multi-feature input tensor.
    - [ ] Stack the `Confirmed`, `Recovered`, and `Death` time series.
    - [ ] Update the GNN layer's input channel dimension from 1 to 3.
- [ ] **Experiment with Advanced GNN Layers**:
    - [ ] Replace the `GCNConv` layer with a `GATConv` (Graph Attention Network) layer to allow the model to learn the importance of different neighbors.
- [ ] **Conduct Hyperparameter Tuning**:
    - [ ] Systematically test different values for `learning_rate`, `lookback_window`, `dropout_rate`, and model dimensions to find the optimal configuration.

### Conceptual and Analytical Depth
- [ ] **Implement LSTM Attention Mechanism**:
    - [ ] [cite_start]As described in the paper[cite: 92, 111], add an attention layer to the LSTM to allow it to weigh the importance of different past time steps when making a forecast.
- [ ] **Perform In-Depth RL Policy Analysis**:
    - [ ] Run the trained agent (with `epsilon=0`) from different starting conditions (e.g., low vs. high initial cases).
    - [ ] Plot the intervention strategies chosen in each scenario to analyze the agent's decision-making process.
- [ ] **Benchmark Against Baseline Policies**:
    - [ ] [cite_start]Compare the RL agent's performance (total infections, cumulative reward) against fixed strategies like "No Intervention" and "Constant Strict Intervention" to quantitatively prove its effectiveness[cite: 214].

---

## 🚀 Future Scope: Extending the Research

[cite_start]This section outlines long-term goals and potential research directions that build upon the current framework, as discussed in the paper's conclusion[cite: 243].

### Data and Feature Expansion
- [ ] **Integrate Dynamic and Heterogeneous Data**:
    - [ ] [cite_start]Incorporate **genomic data** to track viral variants and their impact on transmission rates[cite: 243].
    - [ ] [cite_start]Use **social media data** (e.g., from Twitter) to gauge public sentiment, mobility, and adherence to interventions, providing a real-time behavioral signal[cite: 243].
    - [ ] [cite_start]Include **vaccination campaign data** as a feature to model its effect on the susceptible population[cite: 234].

### Advanced Model Architectures
- [ ] **Explore Hierarchical Reinforcement Learning**:
    - [ ] [cite_start]Develop a hierarchical RL system to manage interventions at different governmental levels (e.g., a high-level agent for national policy and lower-level agents for state or local decisions)[cite: 244]. This would better reflect real-world public health structures.
- [ ] **Develop a Federated Learning Framework**:
    - [ ] [cite_start]Rebuild the training process using federated learning to allow the model to be trained on decentralized data from multiple institutions or countries without compromising data privacy[cite: 245, 259].

### Ethical AI and Interpretability
- [ ] **Implement Model Interpretability Techniques**:
    - [ ] [cite_start]Use methods like **SHAP (SHapley Additive exPlanations)** to explain the GNN-LSTM's predictions and the RL agent's decisions, making the "black box" model more transparent and trustworthy for public health officials[cite: 230].
- [ ] **Conduct Algorithmic Bias and Equity Analysis**:
    - [ ] [cite_start]Perform a post-hoc analysis of the learned RL policy to ensure its recommended interventions do not disproportionately affect vulnerable communities, as discussed in the ethics section of the paper[cite: 226, 227].