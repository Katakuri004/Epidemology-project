"""
Explainability utilities for GNN-LSTM predictions and RL agent decisions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def explain_gnn_attention(model, x: torch.Tensor, norm_adj: torch.Tensor, 
                         target_node: int = 0) -> Dict[str, np.ndarray]:
    """
    Explain GNN-LSTM predictions by analyzing attention patterns and input feature saliency.
    
    Args:
        model: Trained GNNLSTM model
        x: Input tensor (batch, lookback, nodes, features)
        norm_adj: Normalized adjacency matrix
        target_node: Node to explain predictions for
    Returns:
        Dict with attention-like activations, feature importance, and neighbor influence
    """
    model.eval()

    b, t, n, f = x.shape

    # 1) Forward-like pass to extract intermediate activations (no gradients)
    with torch.no_grad():
        x_g = x.reshape(b * t, n, f)
        gnn1_out = model.gnn1.linear(x_g)  # (b*t, n, hidden_gnn)
        gnn1_act = torch.relu(gnn1_out)
        gnn2_out = model.gnn2.linear(gnn1_act)
        gnn2_act = torch.relu(gnn2_out)  # (b*t, n, hidden_gnn)
        gnn_time_node_map = gnn2_act.reshape(b, t, n, -1)[0, :, :, 0].cpu().numpy()  # (t, n)

        x_flat = gnn2_act.reshape(b, t, n, -1).permute(0, 2, 1, 3).contiguous().reshape(b * n, t, -1)
        lstm_out, _ = model.lstm(x_flat)
        # Use softmax over time on a pooled channel dimension as a proxy for temporal attention
        lstm_energy = lstm_out.abs().mean(dim=2)  # (b*n, t)
        lstm_attn = F.softmax(lstm_energy, dim=1).reshape(b, n, t)[0, target_node].cpu().numpy()  # (t,)

    # 2) Gradient-based feature importance (requires grad)
    x_grad = x.clone().detach().requires_grad_(True)
    pred = model(x_grad, norm_adj)  # (b, n, horizon)
    target_scalar = pred[:, target_node, 0].sum()
    model.zero_grad(set_to_none=True)
    if x_grad.grad is not None:
        x_grad.grad.zero_()
    target_scalar.backward()
    feature_grads = x_grad.grad.abs().mean(dim=(0, 1, 2)).detach().cpu().numpy()  # (features,)

    return {
        'gnn_attention': gnn_time_node_map,              # (t, n)
        'lstm_attention': lstm_attn,                     # (t,)
        'feature_importance': feature_grads,            # (features,)
        'target_prediction': pred[0, target_node].detach().cpu().numpy(),  # (horizon,)
        'neighbor_influence': norm_adj[target_node, :].detach().cpu().numpy() if torch.is_tensor(norm_adj) else norm_adj[target_node, :]
    }


def explain_rl_action(agent, state: np.ndarray, action: int) -> Dict[str, np.ndarray]:
    """
    Explain RL agent action using gradients and simple perturbation importance.
    
    Args:
        agent: Trained DQNAgent
        state: Current state vector
        action: Chosen action to explain
    Returns:
        Dict with action explanations and state importance
    """
    agent.q.eval()

    # 1) Gradient w.r.t. state for chosen action
    state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
    state_tensor.requires_grad_(True)
    q_values = agent.q(state_tensor)  # (1, num_actions)
    chosen_q = q_values[0, action]
    agent.q.zero_grad(set_to_none=True)
    if state_tensor.grad is not None:
        state_tensor.grad.zero_()
    chosen_q.backward()
    state_grads = state_tensor.grad.abs().squeeze(0).detach().cpu().numpy()

    # 2) Perturbation-based importance (no gradients needed)
    with torch.no_grad():
        base_q = chosen_q.item()
        state_importance = np.zeros_like(state)
        for i in range(len(state)):
            perturbed = state.copy()
            perturbed[i] = 0.0
            pq = agent.q(torch.from_numpy(perturbed.astype(np.float32)).unsqueeze(0))[0, action].item()
            state_importance[i] = abs(base_q - pq)
        action_conf = F.softmax(q_values, dim=1)[0, action].item()

    return {
        'q_values': q_values.detach()[0].cpu().numpy(),
        'chosen_q': base_q,
        'state_gradients': state_grads,
        'state_importance': state_importance,
        'action_confidence': action_conf,
    }


def plot_gnn_explanation(explanation: Dict[str, np.ndarray], 
                        node_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None):
    """
    Plot GNN-LSTM explanation visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # GNN attention heatmap
    sns.heatmap(explanation['gnn_attention'], 
                ax=axes[0, 0], 
                cmap='viridis',
                xticklabels=node_names if node_names else range(explanation['gnn_attention'].shape[1]),
                yticklabels=range(explanation['gnn_attention'].shape[0]))
    axes[0, 0].set_title('GNN Layer Activation Proxy (Time × Nodes)')
    axes[0, 0].set_xlabel('Nodes')
    axes[0, 0].set_ylabel('Time Steps')

    # LSTM attention over time
    axes[0, 1].plot(explanation['lstm_attention'])
    axes[0, 1].set_title('LSTM Temporal Attention (Proxy)')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Attention Weight')

    # Feature importance
    feature_names = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    feature_names = feature_names[:len(explanation['feature_importance'])]
    axes[1, 0].bar(feature_names, explanation['feature_importance'])
    axes[1, 0].set_title('Input Feature Importance (Grad-based)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Neighbor influence
    if node_names:
        axes[1, 1].bar(node_names, explanation['neighbor_influence'])
    else:
        axes[1, 1].bar(range(len(explanation['neighbor_influence'])), 
                      explanation['neighbor_influence'])
    axes[1, 1].set_title('Neighbor Influence (Adjacency Row)')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rl_explanation(explanation: Dict[str, np.ndarray],
                       state_names: Optional[List[str]] = None,
                       action_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
    """
    Plot RL agent explanation visualizations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Q-values for all actions
    if action_names:
        axes[0, 0].bar(action_names, explanation['q_values'])
    else:
        axes[0, 0].bar(range(len(explanation['q_values'])), explanation['q_values'])
    axes[0, 0].set_title('Q-Values for All Actions')
    axes[0, 0].set_ylabel('Q-Value')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # State importance
    if state_names:
        axes[0, 1].bar(state_names, explanation['state_importance'])
    else:
        axes[0, 1].bar(range(len(explanation['state_importance'])), 
                      explanation['state_importance'])
    axes[0, 1].set_title('State Dimension Importance (Perturbation)')
    axes[0, 1].set_ylabel('Importance Score')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # State gradients
    if state_names:
        axes[1, 0].bar(state_names, explanation['state_gradients'])
    else:
        axes[1, 0].bar(range(len(explanation['state_gradients'])), 
                      explanation['state_gradients'])
    axes[1, 0].set_title('State Gradients (w.r.t Chosen Action)')
    axes[1, 0].set_ylabel('Gradient Magnitude')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Action confidence
    axes[1, 1].text(0.5, 0.5, f"Action Confidence: {explanation['action_confidence']:.3f}", 
                   ha='center', va='center', transform=axes[1, 1].transAxes,
                   fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_title('Action Confidence')
    axes[1, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_explanation_report(model, agent, x: torch.Tensor, norm_adj: torch.Tensor,
                              state: np.ndarray, action: int,
                              node_names: Optional[List[str]] = None,
                              save_dir: str = "explanations") -> Dict[str, str]:
    """
    Generate a comprehensive explanation report for both predictor and RL agent.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Generate explanations
    gnn_explanation = explain_gnn_attention(model, x, norm_adj)
    rl_explanation = explain_rl_action(agent, state, action)

    # Save plots
    gnn_plot_path = os.path.join(save_dir, "gnn_explanation.png")
    rl_plot_path = os.path.join(save_dir, "rl_explanation.png")

    plot_gnn_explanation(gnn_explanation, node_names, gnn_plot_path)
    plot_rl_explanation(rl_explanation, save_path=rl_plot_path)

    return {
        'gnn_plot': gnn_plot_path,
        'rl_plot': rl_plot_path,
        'gnn_explanation': gnn_explanation,
        'rl_explanation': rl_explanation
    }
