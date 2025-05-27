from _keras.trainer import run_keras_train_prediction 
from data.data import denormolize_data 
from pytorch.trainer import run_pytorch_train_prediction 
from utils.evaluation import model_evaluation 
import pandas as pd
import os
import numpy as np
from datetime import datetime as dt  

def generate_loss_value(structure, data, args, keras_models, torch_models):      
    if args.model in keras_models:         
        trues, preds = run_keras_train_prediction(data, structure, args)     
    elif args.model in torch_models:         
        trues, preds = run_pytorch_train_prediction(data, structure, args)     
    else:         
        print('error: incorrect model name. Available models:')         
        print(keras_models)         
        print(torch_models)         
        exit()     
    #3- evaluation models with denormoloized values     
    trues, preds = denormolize_data(trues, preds)     
    
    # Calculate all metrics
    mae, mse, rmse = model_evaluation(trues, preds)
    print(f"Metrics - MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    return mse, trues, preds

def start_job(EXCEL_RESULT_PATH, algorithm, model):
    # Check if the Excel file exists
    if os.path.exists(EXCEL_RESULT_PATH):
        try:
            df_excel = pd.read_excel(EXCEL_RESULT_PATH)
            if len(df_excel) > 0:
                last_job = df_excel.iloc[-1, 0]
                new_job = last_job + 1
            else:
                # File exists but is empty
                new_job = 1
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            new_job = 1
    else:
        # File doesn't exist, so this is the first job
        new_job = 1
        
        # Create directory for results if it doesn't exist
        os.makedirs(os.path.dirname(EXCEL_RESULT_PATH), exist_ok=True)
        
    print(f'job {new_job} : has been started')
    opt_info = str(new_job) + '_' + algorithm + '_' + model      
    now = dt.now()
    timestr = now.strftime("%Y_%m_%d__%H_%M_%S")
    result_path = "./results/" + str(new_job) + "_" + algorithm + timestr + '_' + model + "/"
    
    # Create results directory if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    return new_job, opt_info, result_path  

def show_exp_summary(sol, args, model):
    print(f"Opt: {sol['opt']},"           
          f"Network: {args.model} ,"           
          f"Learning-rate: {sol['learning_rate']}, "           
          f"dropout: {sol['dropout']}, "           
          f"timesteps: {sol['seq_len']}, "           
          f"n-hidden: {sol['n_hidden_units']} ,"           
          f"n-h2: {sol['h2']} ,"           
          f"weight_decay: {sol['weight_decay']}")      
    print("get_parameters")     
    print(model.get_parameters())     
    print(model.get_name())     
    print(model.problem.get_name())
    
    # Print available attributes without trying to find a non-existent solution
    print("Available attributes:", model.get_attributes().keys())
    
    # Access the solution from g_best which is where it's actually stored
    if hasattr(model, 'g_best') and model.g_best is not None:
        print("Best solution found:", model.g_best.solution)
    else:
        print("No g_best attribute or it's None")

def save_model_history_plotting(model, result_path):     
    print(model)     
    ## You can access them all via object "history" like this:     
    model.history.save_global_objectives_chart(filename=result_path + "global_objectives_chart")     
    model.history.save_local_objectives_chart(filename=result_path + "local_objectives_chart/loc")     
    model.history.save_global_best_fitness_chart(filename=result_path + "global_best_fitness_chart/gbfc")     
    model.history.save_runtime_chart(filename=result_path + "runtime_chart/rtc")     
    model.history.save_exploration_exploitation_chart(filename=result_path + "xploration_exploitation_chart/eec")     
    model.history.save_diversity_chart(filename=result_path + "diversity_chart/dc")  

def print_params(structure, itr, args, fitness):
    print(f"best score updated: MSE: ({fitness})")
    print(f"Using optimizer: {structure['opt']} for iteration {itr}")
    print(f"itr {itr} ; MSE: ({fitness}), Paramaters: timesteps: {structure['seq_len']}, "
          f"prediction: {args.pred_len}, model: {args.model}, "
          f"optimizer: {structure['opt']}, h: {structure['n_hidden_units']}, "
          f"lr: {structure['learning_rate']}, dropout: {structure['dropout']}, "
          f"weight_decay: {structure['weight_decay']}, h2: {structure['h2']}, "
          f"features: {structure['features']}")
    print("----\n")
    
def save_model_robustly(model, path):
    """Save model using only NumPy's binary format, completely avoiding pickle and open()"""
    import os
    import numpy as np
    from pathlib import Path
    
    # Create directory - Fix the variable name
    save_dir = path
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    components = []
    
    try:
        # Save parameters dictionary
        if hasattr(model, 'get_parameters'):
            params_dict = model.get_parameters()
            # Convert to a format NumPy can reliably save
            params_list = [(key, value) for key, value in params_dict.items()]
            np.save(os.path.join(save_dir, "parameters.npy"), params_list)
            components.append("parameters")
            success_count += 1
        
        # Save g_best solution and fitness value
        if hasattr(model, 'g_best') and model.g_best is not None:
            if hasattr(model.g_best, 'solution'):
                np.save(os.path.join(save_dir, "best_solution.npy"), model.g_best.solution)
                components.append("best_solution")
                success_count += 1
            
            if hasattr(model.g_best, 'target') and hasattr(model.g_best.target, 'fitness'):
                np.save(os.path.join(save_dir, "best_fitness.npy"), 
                        np.array([model.g_best.target.fitness]))
                components.append("best_fitness")
                success_count += 1
        
        # Save history data if available
        if hasattr(model, 'history'):
            history_data = {}
            
            if hasattr(model.history, 'list_global_best_fit'):
                history_data['global_best_fit'] = model.history.list_global_best_fit
                np.save(os.path.join(save_dir, "global_best_fit.npy"), 
                        np.array(model.history.list_global_best_fit))
                components.append("global_best_fit")
                success_count += 1
            
            if hasattr(model.history, 'list_current_best_fit'):
                np.save(os.path.join(save_dir, "current_best_fit.npy"), 
                        np.array(model.history.list_current_best_fit))
                components.append("current_best_fit") 
                success_count += 1
            
            if hasattr(model.history, 'list_epoch_time'):
                np.save(os.path.join(save_dir, "epoch_times.npy"), 
                        np.array(model.history.list_epoch_time))
                components.append("epoch_times")
                success_count += 1
        
        print(f"Successfully saved {success_count} model components to {save_dir}: {', '.join(components)}")
        return True
        
    except Exception as e:
        print(f"Error saving model with NumPy: {str(e)}")
        return False

def load_best_structure(model_name):
    """
    Load the best structure for a given model from results
    
    Args:
        model_name: Name of the model
        
    Returns:
        Best structure dictionary or None if not found
    """
    best_scores_path = "./results/best_scores.xlsx"
    if not os.path.exists(best_scores_path):
        print(f"Error: {best_scores_path} not found")
        return None
    
    try:
        df = pd.read_excel(best_scores_path)
        # Filter by model name
        model_df = df[df['model'] == model_name]
        
        if len(model_df) == 0:
            print(f"No results found for model {model_name}")
            return None
        
        # Get the row with the lowest MSE
        best_row = model_df.loc[model_df['mse'].idxmin()]
        
        # Create structure dictionary
        structure = {
            'opt': best_row['optimizer'] if 'optimizer' in best_row else 'Adam',
            'learning_rate': best_row['learning_rate'] if 'learning_rate' in best_row else 0.001,
            'dropout': best_row['dropout'] if 'dropout' in best_row else 0.3,
            'seq_len': int(best_row['timesteps']) if 'timesteps' in best_row else 20,
            'n_hidden_units': int(best_row['n_hidden']) if 'n_hidden' in best_row else 64,
            'h2': int(best_row['n_h2']) if 'n_h2' in best_row else 32,
            'weight_decay': best_row['weight_decay'] if 'weight_decay' in best_row else 0.0001,
            'features': best_row['features'].split(',') if isinstance(best_row['features'], str) else ['BRENT Close', 'SENT']
        }
        
        return structure
    except Exception as e:
        print(f"Error loading best structure for {model_name}: {str(e)}")
        return None

def load_model_predictions(model_name, structure, data, args):
    """
    Load or generate predictions for a specific model
    
    Args:
        model_name: Name of the model
        structure: Model structure parameters
        data: Data for training/testing
        args: Arguments
        
    Returns:
        True values and predictions
    """
    # Check if predictions exist in cache
    cache_dir = os.path.join("./results", "predictions_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{model_name}_predictions.npz")
    
    if os.path.exists(cache_file):
        print(f"Loading cached predictions for {model_name}")
        cached_data = np.load(cache_file)
        trues = cached_data['trues']
        preds = cached_data['preds']
    else:
        print(f"Generating predictions for {model_name}")
        # Set the model in args
        args.model = model_name
        
        # Run the appropriate model
        keras_models = ['keras-cnn-bilstm', 'CNN-BiLSTM-att', 'encoder-decoder-LSTM', 'encoder-decoder-GRU']
        torch_models = ['LSTM', 'Bi-LSTM', 'torch-CNN-LSTM', 'Bi-GRU', 'GRU']
        
        if model_name in keras_models:
            trues, preds = run_keras_train_prediction(data, structure, args)
        elif model_name in torch_models:
            trues, preds = run_pytorch_train_prediction(data, structure, args)
        else:
            print(f"Error: Unknown model {model_name}")
            return None, None
        
        # Denormalize data
        trues, preds = denormolize_data(trues, preds)
        
        # Cache the predictions
        np.savez(cache_file, trues=trues, preds=preds)
    
    return trues, preds

def compare_models(models_data, ensemble_metrics=None):
    """
    Compare multiple models and print their performance metrics
    
    Args:
        models_data: Dictionary with model names as keys and (trues, preds) as values
        ensemble_metrics: Optional metrics for ensemble model
    """
    print("\nModel Performance Comparison:")
    print("-" * 60)
    print(f"{'Model':<20} {'MAE':<10} {'MSE':<10} {'RMSE':<10}")
    print("-" * 60)
    
    for model_name, (trues, preds) in models_data.items():
        mae, mse, rmse = model_evaluation(trues, preds)
        print(f"{model_name:<20} {mae:<10.4f} {mse:<10.4f} {rmse:<10.4f}")
    
    if ensemble_metrics:
        print("-" * 60)
        print(f"{'Ensemble':<20} {ensemble_metrics['MAE']:<10.4f} {ensemble_metrics['MSE']:<10.4f} {ensemble_metrics['RMSE']:<10.4f}")
    
    print("-" * 60)