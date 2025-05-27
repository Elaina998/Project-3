import torch
import torch.nn as nn

class TorchBiGRU(nn.Module):
    """
    Bidirectional GRU model for time series forecasting.
    
    This implementation preserves the architecture that has proven successful
    while adding optimization and compatibility features.
    """
    def __init__(self, input_dim, hidden_dim, h2, layer_dim, output_dim, dropout_prob):
        super(TorchBiGRU, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers - keeping the successful configuration
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, 
            dropout=dropout_prob, bidirectional=True
        )
        
        # Simple fully connected layer - as in the successful version
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Get the final time step output
        out = out[:, -1, :]

        # Apply the fully connected layer
        out = self.fc(out)

        return out
    
    def predict(self, x):
        """Method for making predictions with the model in evaluation mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)


class TorchGRU(nn.Module):
    """
    GRU model for time series forecasting.
    
    Note: Fixed hidden state initialization for bidirectional GRU.
    """
    def __init__(self, input_dim, hidden_dim, h2, layer_dim, output_dim, dropout_prob):
        super(TorchGRU, self).__init__()

        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, 
            dropout=dropout_prob, bidirectional=True
        )
        
        # Simple fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # Initializing hidden state - FIXED: using layer_dim * 2 for bidirectional
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Get the final time step output
        out = out[:, -1, :]

        # Apply the fully connected layer
        out = self.fc(out)

        return out
    
    def predict(self, x):
        """Method for making predictions with the model in evaluation mode"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)