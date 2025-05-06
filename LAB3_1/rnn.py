# Model implementation for RNN for Seq-to-Seq regression 
import torch
from torch import nn
import torch.optim as optim
from earlyStopping import EarlyStopping 

class Rnn(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, non_linearity, dropout, bidirectional):
        """
        RNN class for sequence-to-sequence regression.
        
        :param in_size: Input size.
        :param hidden_size: Hidden layer size.
        :param num_layers: Number of RNN layers.
        :param non_linearity: Non-linearity activation function.
        :param dropout: Dropout rate.
        :param bidirectional: If True, use a bidirectional RNN.
        """
        super(Rnn, self).__init__()
        
        self.rnn = nn.RNN(input_size = in_size, 
                          hidden_size = hidden_size, 
                          num_layers = num_layers, 
                          nonlinearity = non_linearity, 
                          dropout = dropout, 
                          bidirectional = bidirectional)
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size, 1)
        else:
            self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, h0):
        """
        Forward pass for the RNN.
        
        :param x: Input tensor.
        :param h0: Initial state value (t=0).
        
        :return: Output tensor and last hidden state.
        """
        out, ht = self.rnn(x, h0)
        out = self.fc(out)
        return out, ht
    
    def get_optimizer(self, optimizer, lr, weight_decay):
        """
        Get the optimizer for the model.
        
        :return: Optimizer.
        """
        optimizer_class = getattr(optim, optimizer)
        return optimizer_class(self.parameters(), lr=lr, weight_decay=weight_decay)
    
    def fit(self, 
            x_train, 
            y_train, 
            x_val, 
            y_val, 
            epochs=10, 
            optimizer='Adam',
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
        optimizer = self.get_optimizer(optimizer, lr, weight_decay)
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStopping(patience=patience, delta=delta)
                
        for epoch in range(epochs):
            h0 = None
            self.train()
            optimizer.zero_grad()
            y_pred, ht = self.forward(x_train, h0)
            loss = loss_fn(y_pred, y_train)
            loss.backward()
            optimizer.step()
            
            train_history
            
            train_history.append(loss.item())
            
           # Validate the model
            self.eval()
            with torch.no_grad():
                y_pred, _ = self(x_val, ht)
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