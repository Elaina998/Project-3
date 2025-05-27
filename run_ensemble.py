import sys
import os
import torch
import numpy as np
import pandas as pd
from config.args import set_args
from data.data import prepare_datat
from utils.helper import start_job, load_best_structure, load_model_predictions, compare_models
from utils.evaluation import model_evaluation, plot_trues_preds
from weighted_ensemble_model import WeightedEnsembleModel
from sklearn.model_selection import train_test_split

def run_ensemble(models_data, args, result_path):
    """
    Run the ensemble model with the given models and data
    
    Args:
        models_data: Dictionary with model names as keys and (trues, preds) as values
        args: Arguments for the ensemble model
        result_path: Path to save results
    
    Returns:
        Ensemble predictions and metrics
    """
    # Extract model names, predictions and true values
    model_names = list(models_data.keys())
    model_predictions = [models_data[model][1] for model in model_names]
    true_values = models_data[model_names[0]][0]  # Assuming all models have same true values
    
    print(f"\nTraining ensemble model with {len(model_names)} base models...")
    print(f"Models included: {', '.join(model_names)}")
    
    # Initialize and run ensemble model
    ensemble = WeightedEnsembleModel(model_names, model_predictions, true_values, result_path)
    
    # You can get validation_split_ratio from args if you add it there
    # e.g., validation_split_ratio = args.ensemble_validation_split if hasattr(args, 'ensemble_validation_split') else 0.2
    validation_split_ratio = getattr(args, 'ensemble_validation_split', 0.2) # Default to 0.2 if not in args

    weights, ensemble_preds, metrics = ensemble.optimize_weights(
        pop_size=args.ensemble_pop_size, 
        iterations=args.ensemble_iterations,
        seed=args.seed, # Assuming GWO constructor or mealpy global seed handles this
        validation_split_ratio=validation_split_ratio
    )
    
    # Print the optimized weights
    print("\nOptimized ensemble weights:")
    for model_name, weight in zip(model_names, weights):
        print(f"{model_name}: {weight:.4f}")
    
    return ensemble_preds, metrics

def main():
    # Parse arguments
    args = set_args()
    
    # Set ensemble flag to True
    args.ensemble = True
    
    # Check if ensemble models are specified
    if not args.ensemble_models:
        print("Error: No ensemble models specified")
        sys.exit(1)
    
    # Initialize
    EXCEL_RESULT_PATH = "./results/ensemble_results.xlsx"
    job_id, running_info, result_path = start_job(EXCEL_RESULT_PATH, "Ensemble", "Weighted")
    
    print("========================================")
    print(f"Starting weighted ensemble model with GWO optimization")
    print(f"Models to include: {args.ensemble_models}")
    print(f"Population size: {args.ensemble_pop_size}")
    print(f"Iterations: {args.ensemble_iterations}")
    print("========================================\n")
    
    # Collect predictions from all models using their best structures
    models_data = {}
    
    for model_name in args.ensemble_models:
        print(f"\nProcessing model: {model_name}")
        
        # Try to load best structure for this model
        best_structure = load_best_structure(model_name)
        
        if best_structure:
            print(f"Using best structure for {model_name}")
            # Prepare data with the best structure's sequence length
            data = prepare_datat(best_structure['seq_len'], args)
            trues, preds = load_model_predictions(model_name, best_structure, data, args)
        else:
            print(f"Using default structure for {model_name}")
            # Use default structure
            seq_len = 20  # Default sequence length
            data = prepare_datat(seq_len, args)
            structure = {
                'opt': 'Adam',
                'learning_rate': 0.001,
                'dropout': 0.3,
                'seq_len': seq_len,
                'n_hidden_units': 64,
                'h2': 32,
                'weight_decay': 0.0001,
                'features': args.features
            }
            trues, preds = load_model_predictions(model_name, structure, data, args)
            
        if trues is not None and preds is not None:
            models_data[model_name] = (trues, preds)
            print(f"Successfully loaded predictions for {model_name}")
            mae, mse, rmse = model_evaluation(trues, preds)
            print(f"Model metrics - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        else:
            print(f"Failed to load predictions for {model_name}")
    
    # Run ensemble model
    if len(models_data) > 1:
        print(f"\nRunning ensemble with {len(models_data)} models...")
        ensemble_preds, metrics = run_ensemble(models_data, args, result_path)
        
        # Save final results
        ensemble_results = {
            'job_id': job_id,
            'Models': ','.join(models_data.keys()),
            'MAE': metrics['MAE'],
            'MSE': metrics['MSE'],
            'RMSE': metrics['RMSE'],
            'num_models': len(models_data)
        }
        
        # Save to Excel
        results_df = pd.DataFrame([ensemble_results])
        if os.path.exists(EXCEL_RESULT_PATH):
            try:
                existing_df = pd.read_excel(EXCEL_RESULT_PATH)
                # Check if columns match
                if set(existing_df.columns) != set(results_df.columns):
                    print("Warning: Column mismatch detected in Excel file. Fixing columns.")
                    existing_df = existing_df.reindex(columns=results_df.columns)
                updated_df = pd.concat([existing_df, results_df], ignore_index=True)
                updated_df.to_excel(EXCEL_RESULT_PATH, index=False)
            except Exception as e:
                print(f"Error updating Excel file: {str(e)}")
                os.rename(EXCEL_RESULT_PATH, f"{EXCEL_RESULT_PATH}.bak")  # Backup the corrupted file
                results_df.to_excel(EXCEL_RESULT_PATH, index=False)
        else:
            results_df.to_excel(EXCEL_RESULT_PATH, index=False)
        
        # Compare models using the helper function
        compare_models(models_data, metrics)
        
        # Plot ensemble results
        plot_trues_preds(
            models_data[list(models_data.keys())[0]][0],  # True values from first model
            ensemble_preds,
            os.path.join(result_path, "ensemble_prediction_plot.jpg")
        )
        
        print("\nEnsemble model completed successfully!")
        print(f"Results saved to {result_path}")
        print(f"Ensemble metrics - MAE: {metrics['MAE']:.4f}, MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        
        # Calculate improvement over best individual model
        best_model_mse = float('inf')
        best_model_name = ""
        
        for model_name, (trues, preds) in models_data.items():
            _, mse, _ = model_evaluation(trues, preds)
            if mse < best_model_mse:
                best_model_mse = mse
                best_model_name = model_name
        
        improvement = (best_model_mse - metrics['MSE']) / best_model_mse * 100
        print(f"\nImprovement over best individual model ({best_model_name}): {improvement:.2f}%")
        
    else:
        print("Error: Not enough models with valid predictions for ensemble")
        print("Need at least 2 models to create an ensemble.")

if __name__ == "__main__":
    main()