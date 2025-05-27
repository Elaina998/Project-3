import numpy as np
import torch
import pandas as pd
import os
import matplotlib.pyplot as plt
from mealpy.swarm_based import GWO
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split # Import train_test_split
from utils.evaluation import model_evaluation, plot_trues_preds
from utils.helper import print_params
from fitness.fitness import save_results
from data.data import denormolize_data

class WeightedEnsembleModel:
    def __init__(self, model_names, model_predictions, true_values, result_path):
        """
        Initialize the Weighted Ensemble Model
        
        Args:
            model_names: List of model names in the ensemble
            model_predictions: List of numpy arrays containing predictions from each model
            true_values: Numpy array of true values
            result_path: Path to save results
        """
        self.model_names = model_names
        self.model_predictions = model_predictions # This is a list of arrays
        self.true_values = true_values
        self.result_path = result_path
        self.weights = None
        self.ensemble_predictions = None
        self.metrics = None
        
    def optimize_weights(self, pop_size=10, iterations=20, seed=42, validation_split_ratio=0.2):
        """
        Optimize the weights using Grey Wolf Optimizer
        
        Args:
            pop_size: Population size for GWO
            iterations: Number of iterations for GWO
            seed: Random seed for reproducibility
            validation_split_ratio: Fraction of data to use for validation during weight optimization.
                                   If 0 or None, uses all data (original behavior).
        """
        print(f"Optimizing ensemble weights using GWO with population={pop_size}, iterations={iterations}")

        if validation_split_ratio and 0 < validation_split_ratio < 1:
            print(f"Using validation split ratio: {validation_split_ratio}")
            # We need to split predictions and true_values consistently.
            # Assuming all prediction arrays and true_values have the same length.
            num_samples = len(self.true_values)
            indices = np.arange(num_samples)
            
            # Split indices to ensure predictions and true values are split consistently
            train_indices, val_indices = train_test_split(indices, test_size=validation_split_ratio, random_state=seed, shuffle=False) # shuffle=False for time series

            opt_true_values = self.true_values[val_indices]
            opt_model_predictions = [pred[val_indices] for pred in self.model_predictions]
            
            print(f"Optimizing weights on {len(opt_true_values)} validation samples.")
        else:
            print("No validation split. Optimizing weights on all provided data.")
            opt_true_values = self.true_values
            opt_model_predictions = self.model_predictions

        # Define the problem for GWO
        class EnsembleProblem:
            def __init__(self, predictions, true_values):
                self.predictions = predictions # List of prediction arrays for the optimization set
                self.true_values = true_values # True values for the optimization set
                self.n_models = len(predictions)
                
            def obj_func(self, weights):
                # Normalize weights to sum to 1
                current_sum = np.sum(weights)
                if current_sum == 0: # Avoid division by zero if all weights are zero
                    # Assign equal weights or handle as an error/special case
                    normalized_weights = np.ones(self.n_models) / self.n_models
                else:
                    normalized_weights = weights / current_sum
                
                # Calculate ensemble prediction
                ensemble_pred = np.zeros_like(self.true_values)
                for i in range(self.n_models):
                    ensemble_pred += normalized_weights[i] * self.predictions[i] # self.predictions are already the opt_model_predictions
                
                # Calculate MSE
                mse = mean_squared_error(self.true_values, ensemble_pred)
                return mse
        
        # Setup the problem with potentially split data
        problem = EnsembleProblem(opt_model_predictions, opt_true_values)
        
        # Setup the bounds (weights between 0 and 1 for each model)
        lb = [0.0] * len(self.model_names) # Use float for bounds
        ub = [1.0] * len(self.model_names) # Use float for bounds
        
        # Initialize GWO optimizer
        # Ensure mealpy's GWO can handle the problem definition correctly
        # The problem object passed to solve should have a 'fitness_function' or similar,
        # or the obj_func itself is passed if the library supports it directly.
        # Assuming optimizer.solve takes the objective function directly:
        optimizer = GWO.OriginalGWO(epoch=iterations, pop_size=pop_size)
        
        # Run the optimization
        # The 'problem' argument to solve in mealpy usually expects an object with lb, ub, obj_func etc.
        # Or, it can take obj_func, lb, ub separately. Your current call seems to be the latter.
        best_position, best_fitness = optimizer.solve(problem.obj_func, lb, ub) # Removed seed from here if GWO sets it internally or if not supported in solve
                                                                                # mealpy typically takes seed in constructor or a global setting.
                                                                                # If GWO.OriginalGWO(..., seed=seed) is possible, use that.
                                                                                # For now, assuming seed is handled by mealpy's global random state or not directly in solve.

        # Normalize the final best_position (weights) to sum to 1
        final_sum = np.sum(best_position)
        if final_sum == 0:
            self.weights = np.ones(len(self.model_names)) / len(self.model_names)
            print("Warning: All optimized weights were zero. Assigning equal weights.")
        else:
            self.weights = best_position / final_sum
        
        print("Optimized weights:")
        for i, (model, weight) in enumerate(zip(self.model_names, self.weights)):
            print(f"{model}: {weight:.4f}")
        
        # Calculate ensemble predictions using the full original dataset
        self.ensemble_predictions = np.zeros_like(self.true_values)
        for i in range(len(self.model_names)):
            self.ensemble_predictions += self.weights[i] * self.model_predictions[i] # Use original self.model_predictions
        
        # Calculate metrics on the full original dataset
        mae, mse, rmse = model_evaluation(self.true_values, self.ensemble_predictions) # Use original self.true_values
        self.metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
        
        print(f"Ensemble Metrics (on full data) - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        # Save results
        self.save_results()
        
        return self.weights, self.ensemble_predictions, self.metrics
    
    def save_results(self):
        """Save ensemble results and visualizations"""
        # Save weights
        weights_df = pd.DataFrame({
            'Model': self.model_names,
            'Weight': self.weights
        })
        weights_df.to_csv(os.path.join(self.result_path, 'ensemble_weights.csv'), index=False)
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Metric': list(self.metrics.keys()),
            'Value': list(self.metrics.values())
        })
        metrics_df.to_csv(os.path.join(self.result_path, 'ensemble_metrics.csv'), index=False)
        
        # Plot true vs predicted
        plot_trues_preds(self.true_values, self.ensemble_predictions, 
                         os.path.join(self.result_path, 'ensemble_predictions.jpg'))
        
        # Save predictions
        save_results(self.true_values, self.ensemble_predictions, self.result_path)
        
        # Plot weights as bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(self.model_names, self.weights)
        plt.title('Ensemble Model Weights')
        plt.xlabel('Model')
        plt.ylabel('Weight')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'ensemble_weights.jpg'))
        plt.close()
        
        return True

def run_ensemble(models_data, args):
    """
    Run the ensemble model with the given models and data
    
    Args:
        models_data: Dictionary with model names as keys and (trues, preds) as values
        args: Arguments for the ensemble model
    
    Returns:
        Ensemble predictions and metrics
    """
    # Create result directory
    result_path = os.path.join("./results", "ensemble_" + "_".join(models_data.keys()))
    os.makedirs(result_path, exist_ok=True)
    
    # Extract model names, predictions and true values
    model_names = list(models_data.keys())
    model_predictions = [models_data[model][1] for model in model_names]
    true_values = models_data[model_names[0]][0]  # Assuming all models have same true values
    
    # Initialize and run ensemble model
    ensemble = WeightedEnsembleModel(model_names, model_predictions, true_values, result_path)
    weights, ensemble_preds, metrics = ensemble.optimize_weights(
        pop_size=args.ensemble_pop_size, 
        iterations=args.ensemble_iterations,
        seed=args.seed
    )
    
    return ensemble_preds, metrics