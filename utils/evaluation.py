from keras.metrics import *
from keras.callbacks import *

# evaluate one or more weekly forecasts against expected values
import matplotlib.pyplot as plt
import traceback
import numpy as np

def model_evaluation(trues, preds):
    """
    Compute multiple metrics between true and predicted values
    Returns MSE, MAE, and RMSE by default for the new code
    """
    try:
        # Get last values if multi-dimensional
        if trues.ndim > 1 and trues.shape[1] > 1:
            true_vals = trues[:, -1]
        else:
            true_vals = trues.flatten()
            
        if preds.ndim > 1 and preds.shape[1] > 1:
            pred_vals = preds[:, -1]
        else:
            pred_vals = preds.flatten()
            
        # Compute metrics
        mae = MAE(pred_vals, true_vals)
        mse = MSE(pred_vals, true_vals)
        rmse = RMSE(pred_vals, true_vals)
        
        # Print metrics for logging
        print(f"Metrics - MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")
        
        # Always return all three metrics needed by the updated code
        return mae, mse, rmse
            
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        traceback.print_exc()
        return float('inf'), float('inf'), float('inf')  # Return infinity for error cases


def plot_loss(history):
     print("plotting")
     plt.plot(history.history['loss'])
     plt.plot(history.history['val_loss'])
     plt.title('model loss')
     plt.ylabel('loss')
     plt.xlabel('epoch')
     plt.legend(['loss', 'val_loss'], loc='upper left')
     plt.show()


def plot_trues_preds(trues, preds, path):
    """Plot true vs predicted values and save to file"""
    plt.figure(figsize=(10, 6))
    
    # Get last values if multi-dimensional
    if trues.ndim > 1 and trues.shape[1] > 1:
        true_vals = trues[:, -1]
    else:
        true_vals = trues.flatten()
        
    if preds.ndim > 1 and preds.shape[1] > 1:
        pred_vals = preds[:, -1]
    else:
        pred_vals = preds.flatten()
        
    plt.plot(true_vals)
    plt.plot(pred_vals)
    plt.title('True vs Predicted Values')
    plt.legend(['True', 'Predicted'], loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def MAE(pred, true):
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    """Calculate Mean Squared Error"""
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    """Calculate Mean Absolute Percentage Error"""
    # Handle division by zero
    mask = true != 0
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))

def MSPE(pred, true):
    """Calculate Mean Squared Percentage Error"""
    # Handle division by zero
    mask = true != 0
    return np.mean(np.square((pred[mask] - true[mask]) / true[mask]))

def metric(pred, true):
    """Legacy metric function for backward compatibility"""
    # Calculate metrics
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    
    print("MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}".format(mae, mse, rmse))
    return mae, mse, rmse