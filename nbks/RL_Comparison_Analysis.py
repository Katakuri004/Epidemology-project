"""
RL Agent Comparison Analysis

This script compares the performance of:
1. Original DQN (from the existing implementation)
2. Enhanced Double Dueling DQN with Prioritized Experience Replay

The comparison demonstrates the improvements in:
- Learning stability
- Intervention policy consistency
- Overall performance
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import joblib
from collections import deque
import random
from typing import List, Tuple
import os

# Import the enhanced components
from Enhanced_RL_Agent_Optimal_Control import (
    PrioritizedReplayBuffer, 
    DuelingQNetwork, 
    DoubleDuelingDQNAgent, 
    EnhancedEpidemicEnv
)

class OriginalQNetwork(nn.Module):
    """Original simple feed-forward network for Q-value approximation."""
    def __init__(self, input_shape, num_actions):
        super(OriginalQNetwork, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_shape), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input state
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class OriginalDQNAgent:
    """Original DQN agent implementation for comparison."""
    
    def __init__(self, state_shape, num_actions, buffer_size=10000, gamma=0.99, lr=1e-4):
        self.num_actions = num_actions
        self.gamma = gamma
        self.q_network = OriginalQNetwork(state_shape, num_actions)
        self.target_network = OriginalQNetwork(state_shape, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Sample a batch from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.cat(states).float()
        actions = torch.tensor(actions).unsqueeze(1).long()
        rewards = torch.tensor(rewards).unsqueeze(1).float()
        next_states = torch.cat(next_states).float()
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).float()

        # Calculate the TD Target
        q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss and update the network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class OriginalEpidemicEnv:
    """Original environment without stability penalty for comparison."""
    
    def __init__(self, predictive_model, initial_state, scaler, edge_index, num_states, forecast_horizon):
        self.model = predictive_model
        self.initial_state = initial_state
        self.scaler = scaler
        self.edge_index = edge_index
        self.num_states = num_states
        self.forecast_horizon = forecast_horizon
        self.state = initial_state.clone().detach()
        self.action_space_n = 3
        self.action_costs = {0: 0.0, 1: 0.4, 2: 1.0}
        self.w_infection = 0.7
        self.w_socioeconomic = 0.3

    def reset(self):
        self.state = self.initial_state.clone().detach()
        return self.state

    def step(self, action):
        # Transition dynamics
        self.model.eval()
        with torch.no_grad():
            predicted_scaled_forecast = self.model(self.state, self.edge_index)
            next_state_scaled_partial = predicted_scaled_forecast[:, 0, :].unsqueeze(1)

        new_state_scaled = torch.cat((self.state[:, 1:, :], next_state_scaled_partial), dim=1)

        # Original reward function (without stability penalty)
        last_day_cases_scaled = self.state[:, -1, :]
        new_day_cases_scaled = next_state_scaled_partial.squeeze(1)

        last_day_cases_log = self.scaler.inverse_transform(last_day_cases_scaled)
        new_day_cases_log = self.scaler.inverse_transform(new_day_cases_scaled)

        last_day_cases_actual = np.expm1(last_day_cases_log)
        new_day_cases_actual = np.expm1(new_day_cases_log)

        delta_infections = np.sum(new_day_cases_actual - last_day_cases_actual)
        normalized_delta_infections = delta_infections / 1_000_000
        intervention_cost = self.action_costs[action]

        # Original reward without stability consideration
        reward = - (self.w_infection * normalized_delta_infections + self.w_socioeconomic * intervention_cost)

        self.state = new_state_scaled
        done = False

        return self.state, reward, done

def train_agent(agent, env, num_episodes, max_steps_per_episode, batch_size, 
                epsilon_start, epsilon_end, epsilon_decay, target_update_freq=10):
    """Train an RL agent and return training metrics."""
    
    epsilon = epsilon_start
    episode_rewards = []
    episode_interventions = []
    episode_stability_scores = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        intervention_sequence = []

        for step in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            agent.learn(batch_size)

            state = next_state
            total_reward += reward
            intervention_sequence.append(action)

        # Calculate stability score
        changes = sum(1 for i in range(1, len(intervention_sequence)) 
                     if intervention_sequence[i] != intervention_sequence[i-1])
        stability_score = 1.0 - (changes / len(intervention_sequence))
        
        episode_rewards.append(total_reward)
        episode_interventions.append(intervention_sequence)
        episode_stability_scores.append(stability_score)
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network for original DQN
        if hasattr(agent, 'update_target_network') and (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}, Stability: {stability_score:.3f}")

    return episode_rewards, episode_interventions, episode_stability_scores

def compare_agents():
    """Compare the performance of original vs enhanced RL agents."""
    
    print("Loading models and parameters...")
    
    # Load model parameters and artifacts
    other_params = torch.load('../models/project_params.pth')
    num_states = other_params['num_states']
    lookback_window = other_params['lookback_window']
    forecast_horizon = other_params['forecast_horizon']
    edge_index = other_params['edge_index']
    X_test_tensor = other_params['X_test_tensor']
    scaler = joblib.load('../models/data_scaler/data_scaler')

    # Load the trained GNN-LSTM model
    class SpatioTemporalGNN_v2(nn.Module):
        def __init__(self, num_nodes, lookback_window, forecast_horizon):
            super(SpatioTemporalGNN_v2, self).__init__()
            self.num_nodes = num_nodes
            self.lookback_window = lookback_window
            self.forecast_horizon = forecast_horizon
            self.gcn = nn.Linear(1, 64)  # Simplified for comparison
            self.dropout1 = nn.Dropout(0.3)
            self.lstm = nn.LSTM(input_size=64 * num_nodes, hidden_size=256, num_layers=2, batch_first=True)
            self.dropout2 = nn.Dropout(0.3)
            self.linear = nn.Linear(256, num_nodes * forecast_horizon)

        def forward(self, x, edge_index):
            gcn_outputs = []
            for t in range(self.lookback_window):
                snapshot = x[:, t, :].unsqueeze(-1)
                batch_gcn_out = [torch.relu(self.gcn(snapshot[i])) for i in range(x.size(0))]
                gcn_out_batch_tensor = torch.stack(batch_gcn_out)
                gcn_outputs.append(gcn_out_batch_tensor)

            gcn_sequence = torch.stack(gcn_outputs, dim=1)
            gcn_sequence = self.dropout1(gcn_sequence)
            lstm_input = gcn_sequence.view(x.size(0), self.lookback_window, -1)
            lstm_out, _ = self.lstm(lstm_input)
            last_time_step_out = lstm_out[:, -1, :]
            last_time_step_out = self.dropout2(last_time_step_out)
            output = self.linear(last_time_step_out)
            output = output.view(-1, self.forecast_horizon, self.num_nodes)
            return output

    # Instantiate and load the trained model
    model_v2 = SpatioTemporalGNN_v2(num_states, lookback_window, forecast_horizon)
    model_v2.load_state_dict(torch.load('../models/gnn_lstm_model.pth'))
    model_v2.eval()

    # Initialize environments and agents
    initial_env_state = X_test_tensor[0].unsqueeze(0)
    
    # Original environment and agent
    original_env = OriginalEpidemicEnv(model_v2, initial_env_state, scaler, edge_index, num_states, forecast_horizon)
    original_agent = OriginalDQNAgent(state_shape=initial_env_state.shape[1:], num_actions=original_env.action_space_n)
    
    # Enhanced environment and agent
    enhanced_env = EnhancedEpidemicEnv(model_v2, initial_env_state, scaler, edge_index, num_states, forecast_horizon)
    enhanced_agent = DoubleDuelingDQNAgent(state_shape=initial_env_state.shape[1:], num_actions=enhanced_env.action_space_n)

    # Training parameters
    num_episodes = 150  # Reduced for faster comparison
    max_steps_per_episode = 100
    batch_size = 32
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995

    print("\n=== Training Original DQN Agent ===")
    original_rewards, original_interventions, original_stability = train_agent(
        original_agent, original_env, num_episodes, max_steps_per_episode, batch_size,
        epsilon_start, epsilon_end, epsilon_decay
    )

    print("\n=== Training Enhanced DQN Agent ===")
    enhanced_rewards, enhanced_interventions, enhanced_stability = train_agent(
        enhanced_agent, enhanced_env, num_episodes, max_steps_per_episode, batch_size,
        epsilon_start, epsilon_end, epsilon_decay
    )

    # Plot comparison results
    plot_comparison_results(original_rewards, original_stability, original_interventions,
                           enhanced_rewards, enhanced_stability, enhanced_interventions)

    # Print comparison summary
    print_comparison_summary(original_rewards, original_stability, original_interventions,
                            enhanced_rewards, enhanced_stability, enhanced_interventions)

def plot_comparison_results(orig_rewards, orig_stability, orig_interventions,
                           enh_rewards, enh_stability, enh_interventions):
    """Plot comprehensive comparison between original and enhanced agents."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot rewards comparison
    ax1.plot(orig_rewards, color='red', alpha=0.7, label='Original DQN')
    ax1.plot(enh_rewards, color='blue', alpha=0.7, label='Enhanced DQN')
    ax1.set_title('Cumulative Reward Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot stability comparison
    ax2.plot(orig_stability, color='red', alpha=0.7, label='Original DQN')
    ax2.plot(enh_stability, color='blue', alpha=0.7, label='Enhanced DQN')
    ax2.set_title('Intervention Stability Comparison')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Stability Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot intervention patterns for last episode
    ax3.plot(orig_interventions[-1], marker='o', color='red', alpha=0.7, label='Original DQN')
    ax3.plot(enh_interventions[-1], marker='s', color='blue', alpha=0.7, label='Enhanced DQN')
    ax3.set_title('Intervention Strategy Comparison (Last Episode)')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Intervention Level')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['None', 'Moderate', 'Strict'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot moving averages for better trend visualization
    window = 10
    orig_rewards_ma = np.convolve(orig_rewards, np.ones(window)/window, mode='valid')
    enh_rewards_ma = np.convolve(enh_rewards, np.ones(window)/window, mode='valid')
    
    ax4.plot(orig_rewards_ma, color='red', alpha=0.7, label='Original DQN (MA)')
    ax4.plot(enh_rewards_ma, color='blue', alpha=0.7, label='Enhanced DQN (MA)')
    ax4.set_title('Moving Average Reward Comparison')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Moving Average Reward')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_comparison_summary(orig_rewards, orig_stability, orig_interventions,
                            enh_rewards, enh_stability, enh_interventions):
    """Print detailed comparison summary."""
    
    print("\n" + "="*60)
    print("RL AGENT COMPARISON SUMMARY")
    print("="*60)
    
    # Calculate metrics
    orig_avg_reward = np.mean(orig_rewards)
    orig_std_reward = np.std(orig_rewards)
    orig_avg_stability = np.mean(orig_stability)
    orig_std_stability = np.std(orig_stability)
    
    enh_avg_reward = np.mean(enh_rewards)
    enh_std_reward = np.std(enh_rewards)
    enh_avg_stability = np.mean(enh_stability)
    enh_std_stability = np.std(enh_stability)
    
    # Calculate improvement percentages
    reward_improvement = ((enh_avg_reward - orig_avg_reward) / abs(orig_avg_reward)) * 100
    stability_improvement = ((enh_avg_stability - orig_avg_stability) / orig_avg_stability) * 100
    
    print(f"\nREWARD PERFORMANCE:")
    print(f"  Original DQN:  {orig_avg_reward:.3f} ± {orig_std_reward:.3f}")
    print(f"  Enhanced DQN:  {enh_avg_reward:.3f} ± {enh_std_reward:.3f}")
    print(f"  Improvement:   {reward_improvement:+.1f}%")
    
    print(f"\nSTABILITY PERFORMANCE:")
    print(f"  Original DQN:  {orig_avg_stability:.3f} ± {orig_std_stability:.3f}")
    print(f"  Enhanced DQN:  {enh_avg_stability:.3f} ± {enh_std_stability:.3f}")
    print(f"  Improvement:   {stability_improvement:+.1f}%")
    
    # Analyze intervention patterns
    def analyze_interventions(interventions):
        all_actions = [action for episode in interventions for action in episode]
        distribution = np.bincount(all_actions, minlength=3)
        return distribution / len(all_actions) * 100
    
    orig_dist = analyze_interventions(orig_interventions)
    enh_dist = analyze_interventions(enh_interventions)
    
    print(f"\nINTERVENTION DISTRIBUTION:")
    print(f"  Original DQN:")
    print(f"    No Intervention:     {orig_dist[0]:.1f}%")
    print(f"    Moderate:            {orig_dist[1]:.1f}%")
    print(f"    Strict:              {orig_dist[2]:.1f}%")
    print(f"  Enhanced DQN:")
    print(f"    No Intervention:     {enh_dist[0]:.1f}%")
    print(f"    Moderate:            {enh_dist[1]:.1f}%")
    print(f"    Strict:              {enh_dist[2]:.1f}%")
    
    print(f"\nKEY IMPROVEMENTS:")
    print(f"  ✓ Double DQN reduces overestimation bias")
    print(f"  ✓ Dueling architecture improves value estimation")
    print(f"  ✓ Prioritized Experience Replay focuses on important experiences")
    print(f"  ✓ Stability penalty encourages consistent policies")
    print(f"  ✓ Enhanced reward function balances multiple objectives")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    compare_agents()
