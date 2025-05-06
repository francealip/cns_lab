# Model implementation for Seq-to-Seq regression

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from earlyStopping import EarlyStopping 
        
class TimeDelayNN(nn.Module):
    def __init__(self,
                 in_dim = 1, 
                 out_dim = 1, 
                 window_sizes = [3, 5, 7],
                 hidden_activations= ['ReLU', 'ReLU', 'ReLU'],
                 hidden_layers = [32, 64, 128],
                 strides = [1, 1, 1],
                 dilations = [1, 2, 3],
                 optimizer = 'Adam'
                 ):
        """
        Time Delay Neural Network (TDNN) class.
        
        :param in_dim: Input channels.
        :param out_dim: Output channels.
        :param window_sizes: Window size values for hidden 1d conv layers
        :param hidden_activations: Activation functions for the hidden layer.
        :param hidden_layers: List of hidden layer dimensions
        :param strides: List of stride values for hidden 1d conv layers
        :param dilations: List of dilation values for hidden 1d conv layers
        :param optimizer: Optimizer for the model.
        """
        assert len(window_sizes) == len(hidden_activations) == len(hidden_layers) == len(strides) == len(dilations), \
            "All parameter lists must have the same length."
        super(TimeDelayNN, self).__init__()
        
        self.optimizer = optimizer
        self.num_layers = len(hidden_layers)

        # Define the hidden layers of the TDNN
        for i in range(self.num_layers):
            if i == 0:
                in_dim = in_dim
            else:
                in_dim = hidden_layers[i-1]
            
            padding = (dilations[i] * (window_sizes[i] - 1)) // 2
            conv_layer = nn.Conv1d(in_channels=in_dim, 
                                   out_channels=hidden_layers[i], 
                                   kernel_size=window_sizes[i], 
                                   stride=strides[i], 
                                   padding=padding,
                                   dilation=dilations[i])
            
            setattr(self, f'conv{i}', conv_layer)
            
            # Add the activation function
            activation = getattr(nn, hidden_activations[i])()
            setattr(self, f'activation{i}', activation)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(hidden_layers[self.num_layers-1], 32)
        self.fc2 = nn.Linear(32, out_dim)
        
    def forward(self, x):
        """
        Forward pass through the TDNN.
        
        :param x: Input tensor.
        :return: Output tensor.
        """
        
        for i in range(self.num_layers):
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'activation{i}')(x) 

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x        
    
    def get_optimizer(self, lr, weight_decay):
        """
        Get the optimizer for the model.
        
        :return: Optimizer.
        """
        optimizer_class = getattr(optim, self.optimizer)
        return optimizer_class(self.parameters(), lr=lr, weight_decay=weight_decay)
        
    def fit(self, 
            x_train, 
            y_train, 
            x_val, 
            y_val, 
            epochs=10, 
            lr=0.001, 
            weight_decay=1e-3, 
            patience=150,
            delta=1e-20,
            verbose=False):
        """
        Fit the model to the data.
        
        :param x_train: Training data.
        :param y_train: Training labels.
        :param x_val: Validation data.
        :param y_val: Validation labels.
        :param epochs: Number of epochs to train.
        :param lr: Learning rate.
        :param weight_decay: Weight decay for the optimizer.
        :param patience: Patience for early stopping.
        :param delta: Minimum change to qualify as an improvement.
        :param verbose: If True, print training progress.
        
        :return: Training and validation loss history.
        """
        train_history, val_history = [], []
        optimizer = self.get_optimizer(lr, weight_decay)
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        
        for epoch in range(epochs):
            # make one step over the training set
            self.train()
            optimizer.zero_grad()
            y_pred = self(x_train)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            train_history.append(loss.item())
            
            # Validate the model
            self.eval()
            with torch.no_grad():
                y_pred = self.forward(x_val)
                loss = loss_fn(y_pred, y_val)
        
                val_history.append(loss.item())
            
            if verbose and (epoch+1)%50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_history[-1]}, Val Loss: {val_history[-1]}")
            
            if early_stopping(val_history[-1], self):
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                self = early_stopping.get_best_model()
                break
            
        return train_history, val_history
        
        
        
        
        
        