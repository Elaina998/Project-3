import argparse
import torch


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parser = argparse.ArgumentParser(description=' Time Series Forecasting')


# data loader

parser.add_argument('--results', type=str, default='./results/', help='location of model results')

# forecasting task
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# model define
parser.add_argument('--n_hidden', type=int, default=4, help='numbers of hidden units')
parser.add_argument('--epoch', type=int, default=1000, help='epoch ')
parser.add_argument('--num_layers', type=int, default=512, help='number of layers')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# optimization
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')#bathsize
parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')#lr


parser.add_argument('--features', type=list, default='', help='list of external variables')
parser.add_argument('--model', type=str, default='LSTM', help='NN model')

# Ensemble parameters
parser.add_argument('--ensemble', type=bool, default=False, help='use ensemble model')
parser.add_argument('--ensemble_models', type=list, 
                    default=['Bi-LSTM', 'Bi-GRU', 'torch-CNN-LSTM', 'LSTM', 'GRU', 
                             'CNN-Bi-LSTM', 'CNN-BiLSTM-Attention', 'Encoder-decoder-LSTM', 'Encoder-decoder-GRU'], 
                    help='models to include in ensemble')
parser.add_argument('--ensemble_pop_size', type=int, default=20, 
                    help='population size for ensemble optimization')
parser.add_argument('--ensemble_iterations', type=int, default=50, 
                    help='iterations for ensemble optimization')
parser.add_argument('--seed', type=int, default=42, help='random seed')


def args_config():
    args = dotdict()

    return args


Algorithms =['GWO', 'L_SHADE']
Features ={0:'BRENT Close',
           10:['SENT', 'BRENT Close'], 
           # Removed feature set 20 that contained 'USDX Price_Difference'
           30:['USDX', 'BRENT Close'],
           # Removed 'USDX Price_Difference' from feature set 120
           120:['SENT', 'BRENT Close'], 130: ['SENT', 'USDX', 'BRENT Close'],
           }
Models ={1:['lstm'], 2:['bi-lstm'], 3:['CNN-lstm'], 4:['CNN-BiLSTM-Attention'],
         5:{'encoder-decoder-lstm'}, 6:['gru'], 7:['bi-gru']}

def set_args():
    args = dotdict()

    #expermint setting:
    args.features= ['BRENT Close', 'SENT']
    args.pred_len = 3
    args.itr = 0


    args.epoch = 2000
    args.batch_size = 32
    args.patience = 30
    args.results = "/results/"
    args.num_layers = 1
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensemble settings
    args.ensemble = False
    args.ensemble_models = ['Bi-LSTM', 'Bi-GRU', 'torch-CNN-LSTM', 'LSTM', 'GRU', 
                         'CNN-Bi-LSTM', 'CNN-BiLSTM-Attention', 'Encoder-decoder-LSTM', 'Encoder-decoder-GRU']
    args.ensemble_pop_size = 20
    args.ensemble_iterations = 50
    args.ensemble_validation_split = 0.2 # Example: Add default validation split
    args.seed = 42
    args.run = None
    args.feature_no = len(args.features)

    return args



