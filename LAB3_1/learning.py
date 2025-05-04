# Learning utilities

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import torch
from timeDelayNN import TimeDelayNN
import random

class EarlyStopping:
    def __init__(self, patience, delta):
        """
        Early stopping class to prevent overfitting.
        
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = float('inf')
        self.stop = False
        self.best_model = None
        
    def __call__(self, val_loss, model):
        """
        Call method to check if training should be stopped.
        
        :param val_loss: Current validation loss.
        :param model: Current model.
        
        :return: True if training should be stopped, False otherwise.
        """
        if val_loss < self.best_score - self.delta:
            self.counter = 0
            self.best_model = model
            self.best_score = val_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
    
    def get_best_model(self):
        return self.best_model   
    

# Utility function to grid search on TDNN

def single_experiment(i, config, x_train, y_train, x_val, y_val, seed=42):
    """
    Run a single experiment with a specific configuration.
    
    :param i: Experiment index.
    :param config: Configurations for the TDNN model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed + i)
    random.seed(seed + i)
    np.random.seed(seed + i)

    # Sample from config1 or config2
    config1, config2 = config
    chosen_config = config1 if torch.rand(1).item() > 0.5 else config2
    params = {k: random.choice(v) for k, v in chosen_config.items()}

    in_dim = x_train.shape[1]
    out_dim = y_train.shape[2]

    tdnn = TimeDelayNN(
        in_dim=in_dim,
        out_dim=out_dim,
        window_sizes=params['window_sizes'],
        hidden_activations=params['hidden_activations'],
        hidden_layers=params['hidden_layers'],
        strides=params['strides'],
        dilations=params['dilations'],
    )

    t_history, v_history = tdnn.fit(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=int(params['epochs'][0]),
        lr=float(params['lr'][0]),
        weight_decay=float(params['weight_decay'][0]),
        patience=int(params['patience'][0]),
        delta=float(params['delta'][0]),
    )

    return {
        'iteration': i,
        'params': params,
        'model': tdnn,
        'train_loss': t_history,
        'val_loss': v_history,
        'final_val_loss': v_history[-1]
    }

def parallel_grid_search(n_iter, config, x_train, y_train, x_val, y_val, seed=42, n_jobs=-1):
    """
    Perform parallel grid search for TDNN hyperparameters.
    
    :param n_iter: Number of iterations for the grid search.
    :param config: Configurations for the TDNN model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param seed: Random seed for reproducibility.
    :param n_jobs: Number of parallel jobs to run.
    
    :return: Best model, training loss history, validation loss history, and parameters.
    """

    results = Parallel(n_jobs=n_jobs)(
        delayed(single_experiment)(i, config, x_train, y_train, x_val, y_val, seed)
        for i in tqdm(range(n_iter), desc="TDNN Grid Search")
    )

    best_result = min(results, key=lambda x: x['final_val_loss'])
    print(f"Best validation loss: {best_result['final_val_loss']}")
    
    return best_result['model'], best_result['train_loss'], best_result['val_loss'], best_result['params']
