import os
import pandas as pd
import numpy as np
from datetime import datetime as dt
from config.args import Features

def decode_solution(solution, encod_data):
    opt_integer = int(solution[0])
    opt = encod_data["OPT_ENCODER"].inverse_transform([opt_integer])[0]
    learning_rate = solution[1]
    n_hidden_units = 2 ** int(solution[2])
    dropout = solution[3]
    seq_len = int(solution[4])
    weight_decay = int(solution[5])
    h2 = int(solution[6])
    
    # Feature selection logic
    # Use the value of h2 to select feature combination
    feature_code = (h2 % 200) // 10 * 10  # Map to one of the keys in Features
    
    # Get the corresponding features
    if feature_code in Features:
        selected_features = Features[feature_code]
    else:
        # Default to a basic feature set if code not found
        selected_features = ['BRENT Close', 'SENT']
    
    # Ensure it's a list
    if not isinstance(selected_features, list):
        selected_features = [selected_features]
    
    # Safety check to remove USDX Price_Difference if it somehow appears
    if 'USDX Price_Difference' in selected_features:
        selected_features.remove('USDX Price_Difference')
    
    return {
        "opt": opt,
        "learning_rate": learning_rate,
        "n_hidden_units": n_hidden_units,
        "dropout": dropout,
        "seq_len": seq_len,
        "weight_decay": weight_decay,
        "h2": h2,
        "features": selected_features
    }
    
def save_results(trues, preds, PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    np.save(PATH + 'preds.npy', preds)
    np.save(PATH + 'vals.npy', trues)

def save_to_file(records, PATH , args):
        df = pd.DataFrame(records)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        PATH= PATH + args.model+".csv"
        print(PATH)
        df.to_csv(PATH, index=False)

def save_to_best_file(excel_path, structure, args, fitness, running_info, running_time, job_id, itr, f=True):
    """
    Save the best model configuration and metrics to an Excel file
    
    Args:
        excel_path: Path to the Excel file
        structure: Model structure parameters
        args: Arguments
        fitness: MSE value (fitness)
        running_info: Information about the run
        running_time: Running time
        job_id: Job ID
        itr: Iteration number
        f: Flag to indicate if this is the final save
    """
    import pandas as pd
    import os
    from utils.evaluation import model_evaluation
    
    # Calculate additional metrics if true values and predictions are available
    mae, rmse = None, None
    if hasattr(args, 'current_trues') and hasattr(args, 'current_preds'):
        mae, mse, rmse = model_evaluation(args.current_trues, args.current_preds)
    
    # Create a dictionary with all the data to save
    data = {
        'job_id': job_id,
        'model': args.model,
        'optimizer': structure['opt'],
        'learning_rate': structure['learning_rate'],
        'dropout': structure['dropout'],
        'timesteps': structure['seq_len'],
        'n_hidden': structure['n_hidden_units'],
        'n_h2': structure['h2'],
        'weight_decay': structure['weight_decay'],
        'features': ','.join(structure['features']),
        'mse': fitness,
        'mae': mae if mae is not None else "N/A",
        'rmse': rmse if rmse is not None else "N/A",
        'running_time': running_time,
        'iteration': itr,
        'info': running_info
    }
    
    # Create a DataFrame with the data
    new_df = pd.DataFrame([data])
    
    # Check if the Excel file exists
    if os.path.exists(excel_path):
        try:
            # Read existing data
            existing_df = pd.read_excel(excel_path)
            
            # Check if columns match, if not, reindex
            if set(existing_df.columns) != set(new_df.columns):
                print("Warning: Column mismatch detected in Excel file. Fixing columns.")
                # Use only the columns from new data to avoid duplicates
                existing_df = existing_df.reindex(columns=new_df.columns)
            
            # Append new data
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Save to Excel
            updated_df.to_excel(excel_path, index=False)
        except Exception as e:
            print(f"Error updating Excel file: {str(e)}")
            # Save as new file if there's an error
            os.rename(excel_path, f"{excel_path}.bak")  # Backup the corrupted file
            new_df.to_excel(excel_path, index=False)
    else:
        # Create new Excel file
        new_df.to_excel(excel_path, index=False)
    
    # Also save to a model-specific CSV file
    model_csv_path = f"./results/{args.model}.csv"
    if os.path.exists(model_csv_path):
        try:
            # Read existing data
            existing_csv_df = pd.read_csv(model_csv_path)
            
            # Check if columns match, if not, reindex
            if set(existing_csv_df.columns) != set(new_df.columns):
                print("Warning: Column mismatch detected in CSV file. Fixing columns.")
                # Use only the columns from new data to avoid duplicates
                existing_csv_df = existing_csv_df.reindex(columns=new_df.columns)
            
            # Append new data
            updated_csv_df = pd.concat([existing_csv_df, new_df], ignore_index=True)
            
            # Save to CSV
            updated_csv_df.to_csv(model_csv_path, index=False)
        except Exception as e:
            print(f"Error updating CSV file: {str(e)}")
            # Save as new file if there's an error
            os.rename(model_csv_path, f"{model_csv_path}.bak")  # Backup the corrupted file
            new_df.to_csv(model_csv_path, index=False)
    else:
        # Create new CSV file
        new_df.to_csv(model_csv_path, index=False)
    
    return True

def register_current_result(score, structure):
    fit_dic ={}

    print("fitness = generate_loss_value(structure, data)")
    fit_dic['score'] = score
    fit_dic['opt'] = structure['opt']
    fit_dic['dropout'] = structure['dropout']
    fit_dic['learning_rate'] = structure['learning_rate']
    fit_dic['n_hidden_units'] = structure['n_hidden_units']
    fit_dic['weight_decay'] = structure['weight_decay']
    fit_dic['h2'] = structure['h2']
    return fit_dic



