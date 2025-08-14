import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import json
from datetime import datetime
import os

# Import our models from the previous implementation
# (Assuming the previous code is in a file called epidemic_models.py)

class EpidemicDataset:
    """Dataset class for handling epidemic time series data"""
    
    def __init__(self, data, sequence_length=30, forecast_horizon=14, scaler=None):
        self.data = data
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler = scaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            # Fit scaler on all data
            reshaped_data = data.reshape(-1, data.shape[-1])
            self.scaler.fit(reshaped_data)
        
        # Normalize data
        self.normalized_data = self._normalize_data(data)
        
        # Create sequences
        self.sequences, self.targets = self._create_sequences()
    
    def _normalize_data(self, data):
        """Normalize the data using fitted scaler"""
        original_shape = data.shape
        reshaped_data = data.reshape(-1, data.shape[-1])
        normalized = self.scaler.transform(reshaped_data)
        return normalized.reshape(original_shape)
    
    def _create_sequences(self):
        """Create input sequences and target sequences"""
        sequences = []
        targets = []
        
        for sample in self.normalized_data:
            for i in range(len(sample) - self.sequence_length - self.forecast_horizon + 1):
                seq = sample[i:i + self.sequence_length]
                target = sample[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon]
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def get_data_loaders(self, batch_size=32, train_split=0.8):
        """Create train and validation data loaders"""
        # Split data
        n_train = int(len(self.sequences) * train_split)
        
        train_sequences = self.sequences[:n_train]
        train_targets = self.targets[:n_train]
        val_sequences = self.sequences[n_train:]
        val_targets = self.targets[n_train:]
        
        # Convert to tensors
        train_dataset = TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.FloatTensor(train_targets)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.FloatTensor(val_targets)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

class ModelTrainer:
    """Comprehensive training pipeline for all models"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def train_hybrid_model(self, model, train_loader, val_loader, epochs=100, lr=0.001):
        """Train the hybrid GNN-LSTM model"""
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training Hybrid Model for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                # For simplified training, we'll use a basic LSTM approach
                # In practice, you'd need proper graph data preparation
                batch_size, seq_len, num_nodes, features = sequences.shape
                
                # Reshape for LSTM processing
                sequences_reshaped = sequences.view(batch_size * num_nodes, seq_len, features)
                targets_reshaped = targets.view(batch_size * num_nodes, -1)
                
                # Simple LSTM forward pass (simplified)
                lstm_model = nn.LSTM(features, 64, 2, batch_first=True)
                fc = nn.Linear(64, targets_reshaped.shape[1])
                
                lstm_out, _ = lstm_model(sequences_reshaped)
                predictions = fc(lstm_out[:, -1, :])
                
                loss = criterion(predictions, targets_reshaped)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)
                    
                    # Same simplified forward pass
                    batch_size, seq_len, num_nodes, features = sequences.shape
                    sequences_reshaped = sequences.view(batch_size * num_nodes, seq_len, features)
                    targets_reshaped = targets.view(batch_size * num_nodes, -1)
                    
                    lstm_out, _ = lstm_model(sequences_reshaped)
                    predictions = fc(lstm_out[:, -1, :])
                    
                    loss = criterion(predictions, targets_reshaped)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_hybrid_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def train_lstm_baseline(self, train_data, val_data, epochs=100):
        """Train standalone LSTM baseline"""
        from epidemic_framework import AttentionLSTM
        
        # Prepare data for LSTM
        train_sequences, train_targets = self._prepare_lstm_data(train_data)
        val_sequences, val_targets = self._prepare_lstm_data(val_data)
        
        # Create model
        input_size = train_sequences.shape[-1]
        model = AttentionLSTM(input_size, hidden_size=64, num_layers=2, 
                             output_size=train_targets.shape[-1])
        model.to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        train_losses = []
        val_losses = []
        
        print(f"Training LSTM Baseline for {epochs} epochs...")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            # Training
            for i in range(0, len(train_sequences), 32):  # Batch size 32
                batch_seq = torch.FloatTensor(train_sequences[i:i+32]).to(self.device)
                batch_target = torch.FloatTensor(train_targets[i:i+32]).to(self.device)
                
                optimizer.zero_grad()
                predictions, _ = model(batch_seq)
                loss = criterion(predictions, batch_target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(val_sequences), 32):
                    batch_seq = torch.FloatTensor(val_sequences[i:i+32]).to(self.device)
                    batch_target = torch.FloatTensor(val_targets[i:i+32]).to(self.device)
                    
                    predictions, _ = model(batch_seq)
                    loss = criterion(predictions, batch_target)
                    val_loss += loss.item()
            
            avg_train_loss = train_loss / (len(train_sequences) // 32)
            avg_val_loss = val_loss / (len(val_sequences) // 32)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        return model, {'train_losses': train_losses, 'val_losses': val_losses}
    
    def _prepare_lstm_data(self, data, sequence_length=30, forecast_horizon=14):
        """Prepare data for LSTM training"""
        sequences = []
        targets = []
        
        for sample in data:
            # Average across nodes for baseline LSTM
            avg_sample = np.mean(sample, axis=1)  # Average over nodes
            
            for i in range(len(avg_sample) - sequence_length - forecast_horizon + 1):
                seq = avg_sample[i:i + sequence_length]
                target = avg_sample[i + sequence_length:i + sequence_length + forecast_horizon]
                sequences.append(seq)
                targets.append(target.flatten())
        
        return np.array(sequences), np.array(targets)

class ModelEvaluator:
    """Comprehensive evaluation of all models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_traditional_seir(self, test_data, num_nodes=10):
        """Evaluate traditional SEIR model"""
        from epidemic_framework import TraditionalSEIRModel
        
        seir_model = TraditionalSEIRModel()
        predictions = []
        actuals = []
        
        for sample in test_data:
            # Use first 30 days for parameter estimation, predict next 14 days
            initial_period = sample[:30]  # First 30 days
            forecast_period = sample[30:44]  # Next 14 days to predict
            
            # Average across nodes
            avg_initial = np.mean(initial_period[-1], axis=0)  # Last day of initial period
            avg_forecast = np.mean(forecast_period, axis=(0,1))  # Average forecast actual
            
            # Simple prediction using last state
            time_points = np.linspace(0, 14, 15)
            prediction = seir_model.simulate(avg_initial, time_points)
            
            predictions.append(prediction[-1, 2])  # Final infected count
            actuals.append(avg_forecast[2])  # Actual infected
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        return {
            'model': 'Traditional SEIR',
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def evaluate_lstm_model(self, model, test_data, scaler):
        """Evaluate LSTM model"""
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sample in test_data:
                # Average across nodes
                avg_sample = np.mean(sample, axis=1)
                normalized_sample = scaler.transform(avg_sample)
                
                # Take sequence for prediction
                sequence = normalized_sample[:30]  # First 30 days
                actual = normalized_sample[30:44]  # Next 14 days
                
                # Make prediction
                seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
                pred, _ = model(seq_tensor)
                pred_np = pred.cpu().numpy().flatten()
                
                # Denormalize
                pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, 4))
                actual_denorm = scaler.inverse_transform(actual)
                
                predictions.append(pred_denorm[:, 2])  # Infected compartment
                actuals.append(actual_denorm[:, 2])
        
        # Flatten for metrics
        pred_flat = np.concatenate(predictions)
        actual_flat = np.concatenate(actuals)
        
        mse = mean_squared_error(actual_flat, pred_flat)
        mae = mean_absolute_error(actual_flat, pred_flat)
        r2 = r2_score(actual_flat, pred_flat)
        
        return {
            'model': 'LSTM Baseline',
            'MSE': mse,
            'MAE': mae,
            'R2': r2,
            'predictions': predictions,
            'actuals': actuals
        }
    
    def evaluate_rl_policy(self, agent, test_data, num_episodes=10):
        """Evaluate RL policy performance"""
        from epidemic_framework import EpidemicEnvironment
        
        total_rewards = []
        total_infections = []
        intervention_costs = []
        
        for episode in range(num_episodes):
            # Random initial state from test data
            initial_state = test_data[np.random.randint(len(test_data))][0]
            
            env = EpidemicEnvironment(None, num_nodes=10, initial_state=initial_state)
            state = env.reset()
            
            episode_reward = 0
            episode_infections = 0
            episode_cost = 0
            
            done = False
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, done = env.step(action)
                
                episode_reward += reward
                episode_infections += np.sum([s[2] for s in env.current_state])  # Count infected
                episode_cost += env.intervention_costs[action]
                
                state = next_state
            
            total_rewards.append(episode_reward)
            total_infections.append(episode_infections)
            intervention_costs.append(episode_cost)
        
        return {
            'model': 'Deep RL Policy',
            'avg_reward': np.mean(total_rewards),
            'avg_infections': np.mean(total_infections),
            'avg_intervention_cost': np.mean(intervention_costs),
            'total_rewards': total_rewards,
            'total_infections': total_infections,
            'intervention_costs': intervention_costs
        }
    
    def compare_models(self, results_list):
        """Create comprehensive comparison of all models"""
        comparison_df = pd.DataFrame()
        
        for result in results_list:
            if 'MSE' in result:  # Forecasting models
                comparison_df = pd.concat([comparison_df, pd.DataFrame([{
                    'Model': result['model'],
                    'MSE': result['MSE'],
                    'MAE': result['MAE'],
                    'R2': result['R2'],
                    'Type': 'Forecasting'
                }])], ignore_index=True)
            else:  # RL model
                comparison_df = pd.concat([comparison_df, pd.DataFrame([{
                    'Model': result['model'],
                    'Avg_Reward': result['avg_reward'],
                    'Avg_Infections': result['avg_infections'],
                    'Avg_Cost': result['avg_intervention_cost'],
                    'Type': 'Control'
                }])], ignore_index=True)
        
        return comparison_df

class ExperimentRunner:
    """Main experiment runner orchestrating the entire research pipeline"""
    
    def __init__(self, output_dir='./results'):
        self.output_dir = output_dir
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def run_complete_experiment(self, data, adjacency_matrix, config=None):
        """Run the complete experimental pipeline"""
        if config is None:
            config = self._get_default_config()
        
        print("="*60)
        print("STARTING COMPREHENSIVE EPIDEMIC MODELING EXPERIMENT")
        print("="*60)
        
        # Split data
        train_data, val_data, test_data = self._split_data(data)
        
        # Prepare datasets
        dataset = EpidemicDataset(train_data)
        train_loader, val_loader = dataset.get_data_loaders()
        
        results = {}
        
        # 1. Train and evaluate LSTM baseline
        print("\n1. Training LSTM Baseline...")
        lstm_model, lstm_history = self.trainer.train_lstm_baseline(
            train_data, val_data, epochs=config['lstm_epochs']
        )
        lstm_results = self.evaluator.evaluate_lstm_model(lstm_model, test_data, dataset.scaler)
        results['LSTM'] = lstm_results
        
        # 2. Evaluate traditional SEIR
        print("\n2. Evaluating Traditional SEIR...")
        seir_results = self.evaluator.evaluate_traditional_seir(test_data)
        results['SEIR'] = seir_results
        
        # 3. Train RL agent
        print("\n3. Training RL Agent...")
        rl_agent = self._train_rl_agent(train_data, config)
        rl_results = self.evaluator.evaluate_rl_policy(rl_agent, test_data)
        results['RL'] = rl_results
        
        # 4. Create comprehensive analysis
        print("\n4. Creating Comprehensive Analysis...")
        analysis = self._create_analysis(results, lstm_history)
        
        # 5. Save results
        self._save_results(results, analysis, config)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return results, analysis
    
    def _get_default_config(self):
        """Get default configuration for experiments"""
        return {
            'lstm_epochs': 100,
            'rl_episodes': 1000,
            'sequence_length': 30,
            'forecast_horizon': 14,
            'train_split': 0.7,
            'val_split': 0.15,
            'test_split': 0.15
        }
    
    def _split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """Split data into train, validation, and test sets"""
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]
        
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        return train_data, val_data, test_data
    
    def _train_rl_agent(self, train_data, config):
        """Train RL agent for intervention optimization"""
        from epidemic_framework import DeepQLearningAgent, EpidemicEnvironment
        
        agent = DeepQLearningAgent(state_dim=40, action_dim=4)
        
        for episode in range(config['rl_episodes']):
            # Random initial state
            initial_state = train_data[np.random.randint(len(train_data))][0]
            env = EpidemicEnvironment(None, num_nodes=10, initial_state=initial_state)
            
            state = env.reset()
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                agent.train()
                state = next_state
            
            if episode % 100 == 0:
                agent.update_target_network()
                print(f"RL Training Episode {episode}, Epsilon: {agent.epsilon:.3f}")
        
        return agent
    
    def _create_analysis(self, results, training_history):
        """Create comprehensive analysis and visualizations"""
        analysis = {}
        
        # Performance comparison
        forecasting_models = ['SEIR', 'LSTM']
        comparison_data = []
        
        for model in forecasting_models:
            if model in results:
                comparison_data.append({
                    'Model': model,
                    'MSE': results[model]['MSE'],
                    'MAE': results[model]['MAE'],
                    'R2': results[model]['R2']
                })
        
        analysis['performance_comparison'] = pd.DataFrame(comparison_data)
        analysis['rl_performance'] = results['RL'] if 'RL' in results else None
        analysis['training_history'] = training_history
        
        return analysis
    
    def _save_results(self, results, analysis, config):
        """Save all results and analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = os.path.join(self.output_dir, f'results_{timestamp}.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for model, result in results.items():
                results_serializable[model] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        results_serializable[model][key] = value.tolist()
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                        results_serializable[model][key] = [v.tolist() for v in value]
                    else:
                        results_serializable[model][key] = value
            
            json.dump(results_serializable, f, indent=2)
        
        # Save performance comparison
        if 'performance_comparison' in analysis:
            comp_file = os.path.join(self.output_dir, f'performance_comparison_{timestamp}.csv')
            analysis['performance_comparison'].to_csv(comp_file, index=False)
        
        # Create visualization
        self._create_visualizations(results, analysis, timestamp)
        
        print(f"Results saved to {self.output_dir}")
    
    def _create_visualizations(self, results, analysis, timestamp):
        """Create comprehensive visualizations"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance comparison
        if 'performance_comparison' in analysis:
            comp_df = analysis['performance_comparison']
            
            # MSE comparison
            axes[0, 0].bar(comp_df['Model'], comp_df['MSE'])
            axes[0, 0].set_title('Model Comparison: MSE')
            axes[0, 0].set_ylabel('Mean Squared Error')
            
            # MAE comparison
            axes[0, 1].bar(comp_df['Model'], comp_df['MAE'])
            axes[0, 1].set_title('Model Comparison: MAE')
            axes[0, 1].set_ylabel('Mean Absolute Error')
            
            # R2 comparison
            axes[1, 0].bar(comp_df['Model'], comp_df['R2'])
            axes[1, 0].set_title('Model Comparison: R²')
            axes[1, 0].set_ylabel('R² Score')
        
        # Training loss (if available)
        if 'training_history' in analysis and analysis['training_history']:
            history = analysis['training_history']
            if 'train_losses' in history:
                axes[1, 1].plot(history['train_losses'], label='Training Loss')
                if 'val_losses' in history:
                    axes[1, 1].plot(history['val_losses'], label='Validation Loss')
                axes[1, 1].set_title('Training History')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
        
        plt.tight_layout()
        viz_file = os.path.join(self.output_dir, f'analysis_{timestamp}.png')
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run the complete experiment"""
    
    # Import our epidemic models
    from epidemic_framework import generate_synthetic_data
    
    print("Generating synthetic epidemic data...")
    data, adjacency_matrix = generate_synthetic_data(
        num_nodes=10, 
        num_days=365, 
        num_samples=100
    )
    
    # Initialize experiment runner
    runner = ExperimentRunner(output_dir='./epidemic_results')
    
    # Run complete experiment
    results, analysis = runner.run_complete_experiment(data, adjacency_matrix)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if 'performance_comparison' in analysis:
        print("\nForecasting Model Performance:")
        print(analysis['performance_comparison'].to_string(index=False))
    
    if 'RL' in results:
        rl_results = results['RL']
        print(f"\nRL Policy Performance:")
        print(f"Average Reward: {rl_results['avg_reward']:.4f}")
        print(f"Average Infections: {rl_results['avg_infections']:.2f}")
        print(f"Average Intervention Cost: {rl_results['avg_intervention_cost']:.4f}")

if __name__ == "__main__":
    main()