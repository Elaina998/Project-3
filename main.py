import sys
import time
import warnings
import numpy as np
import pandas as pd
import os
import torch

from utils.helper import print_params, generate_loss_value, start_job, show_exp_summary, save_model_history_plotting, save_model_robustly
warnings.filterwarnings("ignore")
from config.args import set_args
from mealpy.swarm_based import GWO
from mealpy.utils.problem import Problem
from mealpy.utils.space import FloatVar

from data.data import prepare_datat
from sklearn.preprocessing import LabelEncoder
from utils.evaluation import model_evaluation, plot_trues_preds
from fitness.fitness import decode_solution, save_to_file, save_to_best_file, save_results

# Import only what's needed from ensemble modules
from run_ensemble import main as run_ensemble_main

# Global variables
global_best = {'mse': 1000}
dataset = {}
fitness_list, best_scores = [], []
keras_models = ['keras-cnn-bilstm', 'CNN-BiLSTM-att', 'encoder-decoder-LSTM', 'encoder-decoder-GRU']
torch_models = ['LSTM', 'Bi-LSTM', 'torch-CNN-LSTM', 'Bi-GRU', 'GRU']

class MyProblem(Problem):
    def __init__(self, encod_data, args, excel_path, result_path, job_id, running_info, start_time):
        # Initialize our attributes first, before calling super().__init__
        self.itr = 0
        self.encod_data = encod_data
        self.args = args
        self.excel_path = excel_path
        self.result_path = result_path
        self.job_id = job_id
        self.running_info = running_info
        self.start_time = start_time
        self.global_best = {'mse': 1000, 'trues': None, 'preds': None}
        self.fitness_list = []  # Initialize this BEFORE super().__init__
        
        # Define bounds
        bounds = [
            FloatVar(lb=0, ub=6.99, name="var_0"),
            FloatVar(lb=0.0001, ub=0.01, name="var_1"),
            FloatVar(lb=1, ub=6.99, name="var_2"),
            FloatVar(lb=0.2, ub=0.5, name="var_3"),
            FloatVar(lb=3, ub=29.99, name="var_4"),
            FloatVar(lb=0, ub=0.1, name="var_5"),
            FloatVar(lb=2, ub=127.99, name="var_6")
        ]
        
        # Call the parent class constructor with the bounds
        super().__init__(bounds=bounds, minmax="min", name="HyperparameterOptimization")
    
    def obj_func(self, solution):
        """Override the objective function to compute fitness"""
        # Safety check for initialization
        if not hasattr(self, 'itr') or self.itr is None:
            return 0.0
        
        # Initialize tracking dictionaries if they don't exist
        if not hasattr(self, 'optimizer_counts'):
            self.optimizer_counts = {
                'SGD': 0, 'RMSprop': 0, 'Adagrad': 0, 
                'Adadelta': 0, 'AdamW': 0, 'Adam': 0, 'Adamax': 0
            }
        
        if not hasattr(self, 'feature_counts'):
            self.feature_counts = {
                'BRENT Close': 0, 'SENT': 0, 'USDX': 0, 
                'BRENT Volume': 0,
                'BRENT Price_Difference': 0, 'BRENT Volume_Difference': 0
            }
        
        self.itr += 1
        structure = decode_solution(solution, self.encod_data)
        
        # Log the optimizer and features being tested
        opt_name = structure['opt']
        features = structure['features']
        
        # Update optimizer count
        if opt_name in self.optimizer_counts:
            self.optimizer_counts[opt_name] += 1
        
        # Update feature counts
        for feature in features:
            if feature in self.feature_counts:
                self.feature_counts[feature] += 1
        
        # Log what's being tested in this iteration
        print(f"\n==== Iteration {self.itr} ====")
        print(f"Testing optimizer: {opt_name}")
        print(f"Testing features: {features}")
        print(f"Sequence length: {structure['seq_len']}")
        print(f"Hidden units: {structure['n_hidden_units']}")
        print(f"Learning rate: {structure['learning_rate']}")
        print(f"Dropout: {structure['dropout']}")
        
        # Update args with the selected features
        self.args.features = structure['features']
        
        # Prepare data with the selected features
        data = prepare_datat(structure['seq_len'], self.args)
        self.args.seq_len = structure['seq_len']
        
        # Run the model evaluation
        fitness, trues, preds = generate_loss_value(structure, data, self.args, keras_models, torch_models)
        
        # Store current true values and predictions in args for metrics calculation
        self.args.current_trues = trues
        self.args.current_preds = preds
        
        # Register results
        fit_dic = {"fitness": fitness, "structure": structure}
        self.fitness_list.append(fit_dic)
        
        # Check if we've found a better solution
        if fitness < self.global_best['mse']:
            previous_best = self.global_best['mse']
            improvement = previous_best - fitness
            
            self.global_best['mse'] = fitness
            self.global_best['trues'] = trues
            self.global_best['preds'] = preds
            
            # Save the best model
            current_time = time.time() - self.start_time
            save_to_best_file(self.excel_path, structure, self.args, fitness, self.running_info, 
                            current_time, self.job_id, self.itr, f=False)
            save_results(trues, preds, self.result_path)
            plot_trues_preds(trues, preds, self.result_path + str(fitness) + ".jpg")
            print_params(structure, self.itr, self.args, fitness)
            
            print(f"✓ Improvement found! Previous best: {previous_best}, New best: {fitness}, Difference: {improvement}")
        
        # Print optimizer and feature stats every 10 iterations
        if self.itr % 10 == 0:
            print("\nCumulative optimizer usage so far:")
            for opt, count in self.optimizer_counts.items():
                if count > 0:  # Only show optimizers that were used
                    print(f"  {opt}: {count} times")
            
            print("\nCumulative feature usage so far:")
            for feat, count in self.feature_counts.items():
                if count > 0:  # Only show features that were used
                    print(f"  {feat}: {count} times")
        
        print('Current global best score:', self.global_best['mse'])
        return fitness


def run_single_model(args):
    """Run a single model optimization"""
    iteration, pop_size = 20, 10

    model = GWO.OriginalGWO(epoch=iteration, pop_size=pop_size)
    EXCEL_RESULT_PATH = "./results/best_scores.xlsx"
    job_id, running_info, result_path = start_job(EXCEL_RESULT_PATH, model.__str__(), args.model)

    print("========================================")
    print("start run: ", args.run)
    print(f"algorithm: {model.__str__()} , Model: {args.model} , "
          f" prediction horizon: {args.pred_len} , Features: {args.features}")
    print("=======================================\n")

    # LABEL ENCODER
    encod_data = {}
    OPT_ENCODER = LabelEncoder()
    OPT_ENCODER.fit(['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'AdamW', 'Adam', 'Adamax'])
    encod_data["OPT_ENCODER"] = OPT_ENCODER

    start_time = time.time()
    
    # Create an instance of our custom Problem
    my_problem = MyProblem(
        encod_data=encod_data,
        args=args,
        excel_path=EXCEL_RESULT_PATH,
        result_path=result_path,
        job_id=job_id,
        running_info=running_info,
        start_time=start_time
    )
    
    # Solve using the custom Problem instance
    best_agent = model.solve(my_problem, seed=42)

    end_time = time.time() - start_time

    print("run time:", end_time)
    print(f"Best solution: {best_agent.solution}")
    sol = decode_solution(best_agent.solution, encod_data)

    show_exp_summary(sol, args, model)
    # Save fitness list
    save_to_file(my_problem.fitness_list, result_path, args)
    
    # Save model history
    save_model_robustly(model, os.path.join(result_path, "model_data"))
    save_model_history_plotting(model, result_path)

    # Plot the best results
    if my_problem.global_best['trues'] is not None and my_problem.global_best['preds'] is not None:
        _, mse, _ = model_evaluation(my_problem.global_best['trues'], my_problem.global_best['preds'])
        print("MSE of the best results obtained:", mse)
        plot_trues_preds(my_problem.global_best['trues'], my_problem.global_best['preds'], 
                         result_path + str(mse) + ".jpg")
        save_results(my_problem.global_best['trues'], my_problem.global_best['preds'], result_path)
    
    save_to_best_file(EXCEL_RESULT_PATH, sol, args, best_agent.target.fitness, running_info, 
                      end_time, job_id, my_problem.itr, f=False)
    
    return sol, my_problem.global_best


def run_ensemble_model(args):
    """Run the ensemble model"""
    try:
        print("Running ensemble model...")
        run_ensemble_main()
    except Exception as e:
        print(f"Error running ensemble model: {str(e)}")


if __name__ == '__main__':
    args = set_args()
    
    if len(sys.argv) <= 1:
        print("error: No model name found, you should pass the model name to the main function")
        print("i.e python main.py LSTM")
        print("For ensemble model: python main.py ensemble")
        sys.exit(1)

    # Check for ensemble mode first (simpler condition)
    if sys.argv[1].lower() in ["--run_ensemble", "ensemble"]:
        args.ensemble = True
        run_ensemble_model(args)
    
    # Regular single model mode
    else:
        args.model = sys.argv[1]
        sol, global_best = run_single_model(args)
        
        # Calculate and save final metrics
        if global_best['trues'] is not None and global_best['preds'] is not None:
            mae, mse, rmse = model_evaluation(global_best['trues'], global_best['preds'])
            
            # Update the model-specific CSV with final metrics
            model_csv_path = f"./results/{args.model}.csv"
            if os.path.exists(model_csv_path):
                try:
                    df = pd.read_csv(model_csv_path)
                    # Update the last row with final metrics
                    df.loc[df.index[-1], 'mae'] = mae
                    df.loc[df.index[-1], 'mse'] = mse
                    df.loc[df.index[-1], 'rmse'] = rmse
                    df.to_csv(model_csv_path, index=False)
                except Exception as e:
                    print(f"Error updating final metrics in CSV: {str(e)}")