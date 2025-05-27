import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

from pytorch.models.gru import TorchBiGRU, TorchGRU
from pytorch.models.lstm import TorchBiLSTM, TorchLSTM, TorchCNN_LSTM
from utils.pytorchtools import EarlyStopping

torch.manual_seed(2012)

device = "cuda" if torch.cuda.is_available() else "cpu"


class TorchTrainer:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.scheduler = None

    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        yhat = self.model(x)
        
        # Compute loss
        loss = self.loss_fn(y, yhat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features):
        early_stopping = EarlyStopping(patience=20, verbose=True)
        
        for epoch in range(1, n_epochs + 1):
            # Training phase
            self.model.train()
            batch_losses = []
            
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)
            
            # Validation phase
            self.model.eval()
            batch_val_losses = []
            
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)
                
                # Update learning rate
                if self.scheduler is not None:
                    self.scheduler.step(validation_loss)
                
                # Save best model
                if validation_loss < self.best_val_loss:
                    self.best_val_loss = validation_loss
                    self.best_model_state = self.model.state_dict().copy()
            
            # Early stopping check
            if epoch % 10 == 0:  # Check more frequently
                early_stopping(validation_loss, self.model)
                if early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{n_epochs}")
                print(f"Training Loss: {training_loss:.4f}")
                print(f"Validation Loss: {validation_loss:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

    def evaluate(self, test_loader, batch_size=1, n_features=2):
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                yhat = self.model.predict(x_test) if hasattr(self.model, 'predict') else self.model(x_test)
                yhat = yhat.cpu().data.numpy()
                preds.append(yhat)
                y_test = y_test.cpu().data.numpy()
                trues.append(y_test)
        
        preds = np.array(preds)
        trues = np.array(trues)
        
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        
        return trues, preds

    def plot_losses(self):

        plt.figure(figsize=(4,2))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

def pytorch_data_input(data, args):
    train_features = torch.Tensor(data['X_train'])
    train_targets = torch.Tensor(data['y_train'])
    val_features = torch.Tensor(data['X_valid'])
    val_targets = torch.Tensor(data['y_valid'])
    test_features = torch.Tensor(data['X_test'])
    test_targets = torch.Tensor(data['y_test'])
    '''
    
    print("train_features", train_features.shape )
    print("train_targets", train_targets.shape)

    print("val_features", val_features.shape)
    print("val_targets", val_targets.shape)

    print("test_features", test_features.shape)
    print("test_targets", test_targets.shape)
    '''
    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    # print(train.tensors)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader , test_loader_one

def optimize(model, data, structure, args ):
    train_loader, val_loader, test_loader, test_loader_one = pytorch_data_input(data, args)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = getattr(torch.optim, structure['opt'])(model.parameters(), lr=structure["learning_rate"], weight_decay=structure["weight_decay"])
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=structure["learning_rate"])
    opt = TorchTrainer(model.to(args.device), loss_fn=loss_fn, optimizer=optimizer)

    opt.train(train_loader, val_loader, batch_size=args.batch_size, n_epochs=args.epoch, n_features=data['X_train'].shape[2])
    trues, preds = opt.evaluate(
        test_loader_one,
        batch_size=1,
        n_features=data['X_test'].shape[2])

    return trues, preds

def run_LSTM(data, structure, args):

    if args.model=='LSTM':

        model =TorchLSTM(
                      data['X_train'].shape[2],
                      structure["n_hidden_units"],
                      structure["h2"],
                      args.num_layers, args.pred_len,
                      structure["dropout"])
        trues, preds = optimize(model,data,structure, args)
        print(model.parameters())
        return trues, preds

    elif args.model=='Bi-LSTM':
        model = TorchBiLSTM(
                     data['X_train'].shape[2],
                     structure["n_hidden_units"],
                     structure["h2"],
                     args.num_layers, args.pred_len,
                     structure["dropout"])
        trues, preds = optimize(model,data,structure, args)
        return trues, preds
    elif args.model=='torch-CNN-LSTM':

        model = TorchCNN_LSTM(
                     data['X_train'].shape[2],
                     structure["n_hidden_units"],
                     structure["h2"],
                     args.num_layers, args.pred_len,
                     structure["dropout"])
        trues, preds = optimize(model,data,structure, args)
        return trues, preds


    elif args.model == 'Bi-GRU':
        model = TorchBiGRU(
            data['X_train'].shape[2],
            structure["n_hidden_units"],
            structure["h2"],
            args.num_layers, args.pred_len,
            structure["dropout"])
        trues, preds = optimize(model, data, structure, args)
        return trues, preds

    elif args.model == 'GRU':
        model = TorchGRU(
            data['X_train'].shape[2],
            structure["n_hidden_units"],
            structure["h2"],
            args.num_layers, args.pred_len,
            structure["dropout"])
        trues, preds = optimize(model, data, structure, args)
        return trues, preds



def run_pytorch_train_prediction(data, structure, args):
    trues, preds = run_LSTM(data, structure, args)
    return trues, preds





