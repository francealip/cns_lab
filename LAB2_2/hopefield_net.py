# python file for the Hopfield network class
import torch

class HopfieldNet:
    
    def __init__(self, X):
        """
        Initialize the Hopfield network.
        
        :param size: the size of the patterns to be stored
        :param num_patterns: the number of patterns to be stored
        """
        self.input_patterns = X.clone()
        self.num_patterns, self.size = X.shape
        self.W = self.store_patterns()
        
        
    def store_patterns(self):
        """
        Store patterns in the Hopfield network.
        """
        W = torch.zeros(self.size, self.size)

        for p in self.input_patterns:
            W += torch.outer(p, p)

        W.fill_diagonal_(0) 
        return W / self.size
                  
    def overlap(self, x, target):
        """
        Compute the overlap function between the current state x and the target pattern psi

        :param x: tensor of shape (1, N) - the current state 
        :param target: tensor of shape (1, N) - the target pattern
        """
        return (1/self.size) * (x @ target).item()
        
        
    def energy(self, x):
        """
        Compute the energy of the Hopfield network for a given state x.
        
        :param x: tensor of shape (1, N) - the current state
        """
        return -0.5 * (x @ self.W @ x.t()).item()
        


    def recall(self, input, target_pattern, bias=0.5, max_epochs=10, verbose=False):
        """
        Recall a pattern from the Hopfield network.
        
        :param input: a tensor of shape (size,) containing the pattern to be recalled
        :param target_pattern: the target pattern to be recalled
        :param max_epochs: the maximum number of steps to perform
        :param verbose: if True, print the energy and overlap functions at each step
        
        :return: the recalled pattern
        """
        x = input.clone()
        epoch = 0
        overlaps, energies = [], []
        
        while epoch < max_epochs:
            random_mask = torch.randperm(self.size)
            x_old = x.clone()
            
            for i in random_mask:
                net_input = torch.dot(self.W[i], x) + bias
                x[i] = 1 if net_input > 0 else -1
                
                # compute overlap functions, energy function and flip mask
                overlaps.append(self.overlap(x, target_pattern))
                energies.append(self.energy(x))

            # check if the pattern is stable
            if torch.equal(x, x_old):
                if verbose:
                    print("Pattern is stable at epoch ", epoch, "\n")
                break

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: \n   energy = {energies[epoch]} \n   overlap = {overlaps[epoch]} \n")
            
            epoch+=1
            
        
        return x, energies, overlaps
            