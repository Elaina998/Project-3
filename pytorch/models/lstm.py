# Bidirectional=

import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, h2 ,layer_dim, output_dim, dropout_prob):
        super(TorchLSTM, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc_1 = nn.Linear(hidden_dim, h2)  # fully connected
        self.fc_2 = nn.Linear(h2, output_dim)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out


# Bidirectional=



class TorchBiLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, h2, layer_dim, output_dim, dropout_prob):
        super(TorchBiLSTM, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc_1 = nn.Linear(hidden_dim * 2, h2)  # fully connected
        self.fc_2 = nn.Linear(h2, output_dim)  # fully connected last layer
        self.relu = nn.ReLU()



    def forward(self, x):


        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)  # final output
        return out


class TorchCNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, h2, layer_dim, output_dim, dropout_prob):
        super(TorchCNN_LSTM, self).__init__()
        
        # Model configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.h2 = h2
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.dropout_prob = dropout_prob
        
        # Input normalization
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        
        # First CNN block
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_prob/2)
        )
        
        # Second CNN block
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob/2)
        )
        
        # Adaptive pooling to ensure fixed output size regardless of input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        # Bidirectional LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64,  # Changed from 128 to 64
            hidden_size=hidden_dim, 
            num_layers=layer_dim, 
            dropout=dropout_prob if layer_dim > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Enhanced fully connected layers with residual connections
        self.fc1 = nn.Linear(hidden_dim * 2, h2)
        self.fc2 = nn.Linear(h2, h2)
        self.fc_out = nn.Linear(h2, output_dim)
        
        # Layer normalization for better training stability
        self.layer_norm1 = nn.LayerNorm(hidden_dim * 2)
        self.layer_norm2 = nn.LayerNorm(h2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def attention_mechanism(self, lstm_output):
        """Apply attention mechanism to LSTM output"""
        # lstm_output shape: (batch_size, seq_len, hidden_dim*2)
        attn_weights = self.attention(lstm_output)
        # Apply softmax to get attention weights
        soft_attn_weights = F.softmax(attn_weights, dim=1)
        # Apply attention weights to lstm_output
        context = torch.bmm(soft_attn_weights.transpose(1, 2), lstm_output)
        return context.squeeze(1)
    
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Apply input normalization
        x = self.batch_norm_input(x.transpose(1, 2)).transpose(1, 2)
        
        # CNN feature extraction
        # Reshape for CNN: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        # Apply CNN blocks
        x = self.cnn_block1(x)
        x = self.cnn_block2(x)
        
        # Handle short sequences with adaptive pooling if needed
        if seq_len <= 3:
            # For very short sequences, use adaptive pooling
            x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
            
            # Initialize hidden and cell states
            h0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim, device=x.device).requires_grad_()
            c0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim, device=x.device).requires_grad_()
            
            # LSTM processing
            lstm_out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        else:
            # For longer sequences, use the full CNN pipeline
            # Reshape for LSTM: (batch_size, seq_len, channels)
            lstm_in = x.permute(0, 2, 1)
            
            # Initialize hidden and cell states
            h0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim, device=x.device).requires_grad_()
            c0 = torch.zeros(self.layer_dim * 2, batch_size, self.hidden_dim, device=x.device).requires_grad_()
            
            # LSTM processing
            lstm_out, _ = self.lstm(lstm_in, (h0.detach(), c0.detach()))
        
        # Apply layer normalization
        lstm_out = self.layer_norm1(lstm_out)
        
        # Apply attention mechanism
        attn_out = self.attention_mechanism(lstm_out)
        
        # First fully connected layer
        fc1_out = self.fc1(attn_out)
        fc1_out = F.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Second fully connected layer with residual connection
        fc2_out = self.fc2(fc1_out)
        fc2_out = F.relu(fc2_out)
        fc2_out = self.layer_norm2(fc2_out)
        fc2_out = self.dropout(fc2_out)
        
        # Residual connection
        fc2_out = fc2_out + fc1_out
        
        # Output layer
        out = self.fc_out(fc2_out)
        
        return out
    
    def predict(self, x):
        """Method for making predictions with the model"""
        self.eval()
        with torch.no_grad():
            return self.forward(x)