# Python file for ESN class implementation

import torch

class Esn():
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        hidden_dim, 
        rho, 
        keep_prob, 
        alpha,
        omega_in, 
        omega_bias, 
        scaling_type, 
        washout):
        """
        ESN class implementation for a seq-to-seq regression task
        
        :param in_dim: input feature size
        :param out_dim: output feature size
        :param hidden_dim: hidden neuron size
        :param rho: expected spectral radius for hidden-to-hidden weight matrix
        :param keep_prob: probability of keeping a weight in the hidden-to-hidden matrix
        _param alpha: leaky rate for the reservoir
        :param omega_in: input-to-hidden matrix rescaling parameter 
        :param omega_bias: bias rescaling  parameter
        :param scaling_type: input and bias scaling type
        :param washaout: integere representing number of woshout timesteps
        """
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim 
        self.washout = washout
        self.alpha = alpha
        self.Win = self.init_input_matrix(omega_in, scaling_type)
        self.Wh = self.init_hidden_matrix(rho, keep_prob)
        self.bh = self.init_hiddden_bias(omega_bias, scaling_type)
        Wout = torch.FloatTensor(self.hidden_dim, self.out_dim).uniform_(-1, 1)  
        self.bout = torch.full((1, self.out_dim), torch.randn(1).item())            # Bias for output layer
        self.Wout = torch.cat([Wout, self.bout], dim=0)  # Concatenate Wout and bout to form the output layer
        
    def init_input_matrix(self, omega_in, scaling_type):
        """
        Initialize input to hidden matrix Win
        
        :param omega_in: input-to-hidden matrix rescaling parameter 
        :param scaling_type: input and bias scaling type
        :return: input weight matrix
        """
        Win = torch.FloatTensor(self.hidden_dim, self.in_dim).uniform_(-1, 1)
        if scaling_type == 'range':
          Win = Win * omega_in
        if scaling_type == 'norm':
            Win = Win * (Win/torch.norm(Win))  
        else:
            raise ValueError(f"Unknown dataset: {scaling_type}")
        
        return Win
    
    def init_hidden_matrix(self, rho, keep_prob):
        """
        Initialize input to hidden matrix Win
        
        :param rho: desired spectral radius
        :param keep_prob: probability of keeping a weight in the hidden-to-hidden matrix
        :return: hidden weight matrix
        """
        Wh = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1, 1)
        # sparsify the matrix
        mask = torch.bernoulli(torch.full_like(Wh, keep_prob))
        Wh = Wh * mask
        
        eigenvalues = torch.linalg.eigvals(Wh).abs()
        spectral_radius = torch.max(eigenvalues)
        Wh = Wh * (rho / spectral_radius)
        
        Wh = (Wh - (1 - self.alpha) * torch.eye(self.hidden_dim)) / self.alpha

        
        return Wh
        

    def init_hiddden_bias(self, omega_bias, scaling_type):
        """
        Initialize bias term
        
        :omega_bias: bias rescaling  parameter
        :param scaling_type: input and bias scaling type
        
        :return: bias term
        """
        b = torch.FloatTensor(self.hidden_dim).uniform_(-1, 1)
        if scaling_type == 'range':
          b = b * omega_bias
        if scaling_type == 'norm':
            b = b * (b/torch.norm(b))  
        else:
            raise ValueError(f"Unknown dataset: {scaling_type}")

        return b
    
    def compute_reservoir(self, x):
        """
        Compute hidden activation for the whole sequence
        
        :param x: input sequence tensor (seq_length x feature_size)
        :return: hidden activation tensor (seq_length x hidden_size)
        """
        
        seq_len = x.size(0)
        ht = torch.zeros(self.hidden_dim)
        h = []
        
        with torch.no_grad():
            for t in range(seq_len):
                ht = (1 - self.alpha) * ht + self.alpha * (torch.tanh((self.Wh @ ht) + (self.Win @ x[t]) + self.bh))
                h.append(ht)
            h = torch.stack(h).reshape(seq_len, -1)
        
        return h
            
    def readout(self, h):
        """
        Compute the redout
        
        :param h: hidden activation tensor (seq_len x hidden size)
        :return: output sequence (seq_len x out_dim)
        """
        ones = torch.ones(h.shape[0], 1) 
        h = torch.cat([h, ones], dim=1) 
        y = h @ self.Wout
        return y
    
    def forward(self, x):
        """
        Compute the forward pass
        
        :param x: input sequence tensor (seq_length x feature_size)
        :return: output sequence (seq_length x out_dim)
        """
        h = self.compute_reservoir(x)  
        y = self.readout(h)
        
        return y
    
    def fit(self, x_train, y_train, lambd):
        h = self.compute_reservoir(x_train)  
        
        # Washout if needed
        h = h[self.washout:]
        x_train = x_train[self.washout:]
        y_train = y_train[self.washout:]
        
        # Add bias term
        ones = torch.ones(h.shape[0], 1)
        h = torch.cat([h, ones], dim=1) 

        if lambd == 0:
            self.Wout = torch.linalg.pinv(h) @ y_train
        else:
            I = torch.eye(h.shape[1])
            self.Wout = torch.linalg.pinv(h.T @ h + lambd * I) @ h.T @ y_train
        
    def loss(self, y_pred, y_true):
        """
        Compute the loss function
        
        :param y_pred: predicted output
        :param y_true: true output
        :return: MSE loss value
        """
        return torch.nn.functional.mse_loss(y_pred, y_true)