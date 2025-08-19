# Epidemiology Forecasting and Optimal Control Project

## Overview

This project implements a comprehensive system for epidemic forecasting and optimal intervention control using advanced machine learning techniques. The system combines a **Graph Neural Network-LSTM (GNN-LSTM)** predictive model with an **enhanced Reinforcement Learning (RL)** agent to provide both accurate epidemic forecasts and optimal intervention strategies.

## 🎯 Project Goals

- **Predictive Modeling**: Forecast epidemic spread across multiple regions using spatiotemporal data
- **Optimal Control**: Determine the best intervention strategies to minimize infections while considering socioeconomic costs
- **Policy Stability**: Ensure intervention policies are consistent and practical for real-world implementation
- **Scalability**: Handle multi-regional epidemic data with spatial dependencies

## 🏗️ Architecture

### 1. Predictive Model (GNN-LSTM)
- **Graph Neural Networks**: Capture spatial dependencies between regions
- **LSTM Networks**: Model temporal patterns in epidemic spread
- **Multi-feature Support**: Handle confirmed cases, recoveries, and deaths
- **Attention Mechanisms**: Focus on relevant spatial and temporal patterns

### 2. Control Model (Enhanced RL Agent)
- **Double DQN**: Reduces overestimation bias in Q-value estimation
- **Dueling DQN**: Separates state value and action advantage learning
- **Prioritized Experience Replay**: Focuses learning on surprising experiences
- **Enhanced Reward Function**: Balances infection control, costs, and policy stability

## 📁 Project Structure

```
code/
├── assets/                          # Generated plots and visualizations
│   ├── lstm-loss.png
│   ├── lstm-result.png
│   └── RL-result.png
├── gnn-lstm-params/                 # Model parameters and artifacts
│   ├── gnn_lstm_model_v2.pth
│   ├── min_max_scaler
│   └── other_params.pth
├── md/                              # Documentation
│   ├── To-Do.md                     # Project roadmap and tasks
│   └── Enhanced_RL_Documentation.md # Detailed RL implementation guide
├── models/                          # Trained models and data
│   ├── data_scaler/
│   ├── gnn_lstm_model.pth
│   ├── project_params.pth
│   └── rl_agent_q_network.pth
├── nbks/                            # Jupyter notebooks
│   ├── basemodel.ipynb              # Basic model implementation
│   ├── GNN_LSTM_MODEL.ipynb         # GNN-LSTM predictive model
│   └── Deep_Reinforcement_Learning_(RL)_agent_for_optimal_control.ipynb
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Key Features

### Predictive Model Enhancements
- **Spatiotemporal Learning**: Combines graph neural networks with LSTM for spatial and temporal modeling
- **Multi-feature Input**: Supports multiple epidemic indicators (cases, recoveries, deaths)
- **Attention Mechanisms**: Focuses on relevant patterns in both space and time
- **Robust Training**: Includes dropout layers and regularization for better generalization

### RL Agent Improvements
- **Double DQN**: Eliminates overestimation bias in Q-value estimation
- **Dueling Architecture**: Separates value and advantage streams for better learning
- **Prioritized Experience Replay**: Efficiently learns from important experiences
- **Stability Penalty**: Encourages consistent intervention policies
- **Enhanced Reward Function**: Multi-objective optimization considering:
  - Infection control (60% weight)
  - Socioeconomic costs (30% weight)
  - Policy stability (10% weight)

## 📊 Results and Performance

### Predictive Model Performance
- **Accuracy**: High forecasting accuracy across multiple regions
- **Spatial Dependencies**: Successfully captures inter-regional transmission patterns
- **Temporal Patterns**: Learns complex temporal dynamics of epidemic spread

### RL Agent Performance
- **Learning Stability**: Reduced variance in training curves
- **Policy Consistency**: More stable intervention strategies
- **Efficiency**: Faster convergence through prioritized experience replay
- **Balance**: Optimal trade-off between infection control and intervention costs

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy, Pandas, Matplotlib
- Scikit-learn

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd epidemiology-Project/code

# Install dependencies
pip install -r requirements.txt

# For PyTorch Geometric (if needed)
pip install torch-geometric
```

## 📖 Usage

### 1. Training the Predictive Model
```python
# Run the GNN-LSTM training notebook
jupyter notebook nbks/GNN_LSTM_MODEL.ipynb
```

### 2. Training the RL Agent
```python
# Run the enhanced RL agent training notebook
jupyter notebook nbks/Deep_Reinforcement_Learning_(RL)_agent_for_optimal_control.ipynb
```

### 3. Model Evaluation
```python
# Load and evaluate trained models
import torch
import joblib

# Load predictive model
model = SpatioTemporalGNN_v2(num_nodes, lookback_window, forecast_horizon)
model.load_state_dict(torch.load('models/gnn_lstm_model.pth'))

# Load RL agent
agent = DoubleDuelingDQNAgent(state_shape, num_actions)
agent.q_network.load_state_dict(torch.load('models/rl_agent_q_network.pth'))
```

## 🔧 Technical Specifications

### GNN-LSTM Model
- **GCN Layers**: Graph Convolutional Networks for spatial modeling
- **LSTM Layers**: 2-layer LSTM with 256 hidden units
- **Dropout**: 0.3 dropout rate for regularization
- **Input Features**: Multi-dimensional time series data
- **Output**: Multi-step forecasts for all regions

### Enhanced RL Agent
- **Network Architecture**: Dueling DQN with shared feature layers
- **Replay Buffer**: 10,000 experience capacity with prioritized sampling
- **Learning Rate**: 1e-4 with Adam optimizer
- **Target Update**: Soft updates with τ = 0.01
- **Gradient Clipping**: Max norm of 1.0 for training stability

### Hyperparameters
| Component | Parameter | Value |
|-----------|-----------|-------|
| **GNN-LSTM** | Learning Rate | 1e-3 |
| | Lookback Window | 14 days |
| | Forecast Horizon | 7 days |
| | Hidden Size | 256 |
| **RL Agent** | Buffer Size | 10,000 |
| | Alpha (PER) | 0.6 |
| | Beta (PER) | 0.4 |
| | Target Update Freq | 100 |

## 📈 Performance Metrics

### Predictive Model Metrics
- **Mean Absolute Error (MAE)**: Measures forecast accuracy
- **Root Mean Square Error (RMSE)**: Penalizes large errors
- **Spatial Correlation**: Captures inter-regional dependencies

### RL Agent Metrics
- **Cumulative Reward**: Overall performance across episodes
- **Stability Score**: Consistency of intervention policies
- **Intervention Distribution**: Frequency of different intervention levels
- **Reward-Stability Correlation**: Relationship between performance and consistency

## 🔬 Research Contributions

### Novel Approaches
1. **Spatiotemporal GNN-LSTM**: Combines graph neural networks with LSTM for epidemic forecasting
2. **Enhanced RL for Epidemic Control**: Multi-objective optimization with stability constraints
3. **Prioritized Experience Replay**: Efficient learning from important experiences
4. **Stability-Aware Reward Function**: Encourages practical intervention policies

### Technical Innovations
- **Double DQN for Epidemic Control**: Reduces overestimation bias in intervention decisions
- **Dueling Architecture**: Better separation of state value and action advantages
- **Multi-feature Spatiotemporal Modeling**: Comprehensive epidemic data representation
- **Stability Penalty Mechanism**: Ensures consistent policy recommendations

## 🚧 Future Enhancements

### Planned Improvements
- [ ] **Graph Attention Networks**: Replace GCN with GAT for adaptive neighbor weighting
- [ ] **Multi-Agent RL**: Hierarchical control for different governmental levels
- [ ] **Federated Learning**: Privacy-preserving distributed training
- [ ] **Meta-Learning**: Adaptation to new epidemic scenarios
- [ ] **Interpretability Tools**: SHAP analysis for decision explanations

### Research Directions
- **Dynamic Graph Construction**: Adaptive spatial dependency modeling
- **Multi-Objective Optimization**: Advanced techniques for balancing multiple objectives
- **Real-time Adaptation**: Online learning for changing epidemic conditions
- **Ethical AI**: Bias detection and fairness in intervention recommendations

## 📚 References

### Key Papers
- Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double q-learning
- Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning
- Schaul, T., et al. (2016). Prioritized experience replay
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks

### Related Work
- Epidemic forecasting with deep learning
- Reinforcement learning for public health
- Graph neural networks for spatiotemporal data
- Multi-objective optimization in healthcare

## 🤝 Contributing

We welcome contributions to improve the project! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Issue reporting

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions, issues, or collaboration opportunities:
- **Issues**: Please use the GitHub issue tracker
- **Discussions**: Join our community discussions
- **Email**: [Project Contact Email]

## 🙏 Acknowledgments

- **Data Sources**: Public health agencies and data providers
- **Research Community**: Contributors to open-source ML libraries
- **Academic Support**: Research institutions and collaborators
- **Open Source**: PyTorch, PyTorch Geometric, and other open-source tools

---

**Note**: This project is for research purposes. Please consult with public health experts before using these models for real-world decision-making.