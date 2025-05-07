# Python file for ESN class implementation

import torch

class Esn():
    def __init__(self, in_dim, out_dim, hidden_dim, rho, omega_in, omega_bias, scaling_type, washout):
        """
        ESN class implementation for a seq-to-seq regression task
        
        :param in_dim: input feature size
        :param out_dim: output feature size
        :param hidden_dim: hidden neuron size
        :param rho: expected spectral radius for hidden-to-hidden weight matrix
        :param omega_in: input-to-hidden matrix rescaling parameter 
        :param omega_bias: bias rescaling  parameter
        :param scaling_type: input and bias scaling type
        :param washaout: integere representing number of woshout timesteps
        """
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim 
        self.washout = washout
        self.Win = self.init_input_matrix(omega_in, scaling_type)
        self.Wh = self.init_hidden_matrix(rho)
        self.bh = self.init_hiddden_bias(omega_bias, scaling_type)
        self.Wout = torch.FloatTensor(self.hidden_dim + 1, self.out_dim).uniform_(-1, 1)  # +1 to include bias
        
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
    
    def init_hidden_matrix(self, rho):
        """
        Initialize input to hidden matrix Win
        
        :param rho: desired spectral radius
        :return: hidden weight matrix
        """
        Wh = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1, 1)
        eigenvalues = torch.linalg.eigvals(Wh).abs()
        spectral_radius = torch.max(eigenvalues)
        Wh = Wh * (rho / spectral_radius)
        
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
        ht_minus_1 = torch.zeros(self.hidden_dim)
        h = []
        
        for t in range(seq_len):
            ht = torch.tanh((self.Wh @ ht_minus_1) + (self.Win @ x[t]) + self.bh)
            h.append(ht)
            ht_minus_1 = ht 
        h = torch.stack(h).reshape(seq_len, -1)
        
        return h[self.washout:]
            
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
    
    def fit(self, x_train, y_train, lambd):
        h = self.compute_reservoir(x_train)  
        
        # Add bias term
        ones = torch.ones(h.shape[0], 1)
        h = torch.cat([h, ones], dim=1) 

        Ht = h.T
        I = torch.eye(Ht.shape[0])
        self.Wout = torch.linalg.solve(Ht @ h + lambd * I, Ht @ y_train)