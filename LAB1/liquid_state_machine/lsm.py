# This file contains the Liquid State Machine class and the MLP class used in the Liquid State Machine

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def predict(self, x):
        with torch.no_grad():  
            return self.forward(x)
    
    def fit(self, X, y, epochs=50, learning_rate=0.001, batch_size=32, verbose=False):
        """
        Trains the model using mini-batches.
        
        :param X: Input data (torch.Tensor).
        :param y: Target (torch.Tensor).
        :param num_epochs: Total number of epochs.
        :param learning_rate: Learning rate for the optimizer.
        :param batch_size: Mini-batch size.
        :param verbose: Flag to print the loss progression.
        """
        # Create a dataset and a DataLoader to handle mini-batches
        dataset = TensorDataset(X, y.unsqueeze(1))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss and optimizer
        criterion = nn.L1Loss()  
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and weight update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Average loss for the epoch
            epoch_loss /= len(dataloader)
            if verbose and (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')
                

class LSM:
    
    def __init__(self, n=1000, alpha=0.8, Win_e=5, Win_i=2, W_e=0.5, W_i=1, 
                 scaler=None, redout='linear', hidden_size=50):
        """
        Liquid State Machine class initialization
        
        :param n: number of neurons in the liquid state
        :param alpha: proportion of excitatory neurons
        :param Win_e: input weights to excitatory neurons
        :param Win_i: input to inhibitory neurons
        :param W_e: excitatory to excitatory neurons
        :param W_i: inhibitory to excitatory neurons

        """
        self.n = n
        self.alpha = alpha
        self.Ne = int(n * alpha)
        self.Ni = n - self.Ne
        
        self.scaler = scaler
        self.U = np.concatenate((Win_e*np.ones(self.Ne), Win_i*np.ones(self.Ni)))
        self.S = np.concatenate((
            W_e*np.random.rand(self.Ne+self.Ni, self.Ne), 
            -W_i*np.random.rand(self.Ne+self.Ni, self.Ni)), axis=1)
        
        self.redout = redout
        
        if redout == 'linear':
            self.w_out = None
        else:
            self.mlp = MLP(n, hidden_size, 1)
            
        self.states = []
        self.firings = []
    
    def compute_liquid_state(self, input, verbose=False):
        """
        Liquid State Machine computation
        
        :param input: input data
        :param verbose: plot the firings
        
        :return: output of the LSM
        """     
        re = np.random.rand(self.Ne)
        ri = np.random.rand(self.Ni)
        
        a = np.concatenate((0.015*np.ones(self.Ne), 0.015+0.08*ri)) #provare 0.1 e 0.001
        b = np.concatenate((0.2*np.ones(self.Ne), 0.25-0.05*ri))
        
        
        # EXC Tonic Spiking parameters
        c = np.concatenate((-65+15*re**2, -65*np.ones(self.Ni)))
        d = np.concatenate((8-6*re**2, 2*np.ones(self.Ni)))
        """
        # EXC Tonic Bursting parameters
        c = np.concatenate((-50+15*re**2, -65*np.ones(self.Ni)))
        d = np.concatenate((2*np.ones(self.Ne), 8-6*ri**2))
        """
        
        
        v = -65*np.ones(self.Ne+self.Ni)    # Initial values of v
        u = b*v                             # Initial values of u
        firings = []                        # spike timings
        states = []                         # here we construct the matrix of reservoir states

        #input = self.scaler.transform(input.reshape(-1, 1)).squeeze()
        
        for t in range(len(input)):
            # we don't need random thalamic input:
            # I = np.concatenate((5*np.random.randn(Ne), 2*np.random.randn(Ni)))  # thalamic input
            # we use instead the input from the external time series!
            I = input[t] * self.U
            fired = np.where(v >= 30)[0]  # indices of spikes
            firings.append(np.column_stack((t+np.zeros_like(fired), fired)))
            v[fired] = c[fired]
            u[fired] = u[fired] + d[fired]
            I = I + np.sum(self.S[:, fired], axis=1)
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # step 0.5 ms
            v = v + 0.5*(0.04*v**2 + 5*v + 140 - u + I)  # for numerical stability
            u = u + a*(b*v - u)
            states.append(v >= 30)

        firings = np.concatenate(firings)
        
        if verbose:
            plt.figure(figsize=[10,7])
            plt.title('Firings')
            plt.plot(firings[:, 0], firings[:, 1], '.')
        
        states = np.array(states, dtype=int)
        return states, firings

    def fit(self, x, y, lambda_reg=0, lr=0.001, epochs=50, batch_size=120, verbose=False):
        """
        Liquid State Machine linear redout training
        
        :param X: input data
        :param y: output data
        :param lambda_reg: regularization parameter
        :param lr: learning rate
        :param epochs: number of epochs
        :param batch_size: batch size
        :param verbose: alloes printing
        """
        if self.redout == 'linear':
            self.fit_linear_readout(x, y, lambda_reg, verbose)
        else:
            self.fit_mlp_readout(x, y, lr, epochs, batch_size, verbose)

    
    def fit_mlp_readout(self, x, y, lr, epochs, batch_size, verbose=False):
        """
        Liquid State Machine MLP redout training
        """
        self.states, self.firings = self.compute_liquid_state(x, verbose)
        #y = self.scaler.transform(y.reshape(-1, 1)).squeeze()
        X = self.states
        y = torch.tensor(y, dtype=torch.float32)
        X = torch.tensor(X, dtype=torch.float32)
        
        self.mlp.fit(X, y, epochs=epochs, learning_rate=lr, batch_size=batch_size, verbose=verbose)
        return
    
    def fit_linear_readout(self, x, y, lambda_reg, verbose=False):
        """
        Liquid State Machine linear redout training
        """
        self.states, self.firings = self.compute_liquid_state(x, verbose)
        #y = self.scaler.transform(y.reshape(-1, 1)).squeeze()
        if lambda_reg != 0:
            I = np.eye(self.states.shape[1])   
            self.w_out = np.linalg.inv(self.states.T @ self.states + lambda_reg * I) @ self.states.T @ y
        else:
            self.w_out = np.linalg.pinv(self.states) @ y
            
        return
    
    def predict(self, x, states=None):
        """
        Liquid State Machine prediction
        
        :param x: input data
        :param states: liquid state
        :return: predicted output
        """
        if states is None:
            states, _ = self.compute_liquid_state(x)
        
        if self.redout == 'linear':
            y_pred = states @ self.w_out
            #y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
        else:
            X = torch.tensor(states, dtype=torch.float32)
            y_pred = self.mlp.predict(X)
            y_pred = y_pred.detach().numpy().squeeze()
            #y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).squeeze()
        
        return y_pred
    
    
    def evaluate(self, y, y_pred):
        """
        Liquid State Machine evaluation
        
        :param y: true output
        :param y_pred: predicted output
        :return: mean absolute error
        """
        y_pred = torch.tensor(y_pred, dtype=torch.float32).squeeze()
        y = torch.tensor(y, dtype=torch.float32).squeeze()
        
        return nn.L1Loss()(y_pred, y).item()