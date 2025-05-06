# Learning utilities

from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import torch
from timeDelayNN import TimeDelayNN
from rnn import Rnn
import random
    
# Utility function to grid search on TDNN

def tdnn_single_experiment(i, config, x_train, y_train, x_val, y_val, seed=42):
    """
    Run a single experiment with a specific configuration with TDNN.
    
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

    tdnn = TimeDelayNN(
        in_dim = x_train.shape[1],
        out_dim = y_train.shape[2],
        window_sizes = params['window_sizes'],
        hidden_activations = params['hidden_activations'],
        hidden_layers = params['hidden_layers'],
        strides = params['strides'],
        dilations = params['dilations'],
    )

    t_history, v_history = tdnn.fit(
        x_train = x_train,
        y_train = y_train,
        x_val = x_val,
        y_val = y_val,
        epochs = int(params['epochs'][0]),
        lr = float(params['lr'][0]),
        weight_decay = float(params['weight_decay'][0]),
        patience = int(params['patience'][0]),
        delta = float(params['delta'][0]),
    )

    return {
        'iteration': i,
        'params': params,
        'model': tdnn,
        'train_loss': t_history,
        'val_loss': v_history,
        'final_val_loss': v_history[-1]
    }
    
def rnn_single_experiment(i, config, x_train, y_train, x_val, y_val, seed=42):
    """
    Run a single experiment with a specific configuration with RNN.
    
    :param i: Experiment index.
    :param config: Configurations for the RNN model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param seed: Random seed for reproducibility.
    """
    torch.manual_seed(seed + i)
    random.seed(seed + i)
    np.random.seed(seed + i)

    params = {k: random.choice(v) for k, v in config.items()}
    
    if params['dropout'] != 0 and params['num_layers'] == 1:
        params['num_layers'] = 2
        
    rnn = Rnn(
        in_size = x_train.shape[1],
        hidden_size = params['hidden_size'],
        num_layers = params['num_layers'],
        non_linearity = params['non_linearity'],
        dropout = params['dropout'],
        bidirectional = params['bidirectional'],
    )

    t_history, v_history = rnn.fit(
        x_train = x_train,
        y_train = y_train,
        x_val =  x_val,
        y_val = y_val,
        epochs = int(params['epochs']),
        lr = float(params['lr']),
        weight_decay = float(params['weight_decay']),
        patience = int(params['patience']),
        delta = float(params['delta']),
    )

    return {
        'iteration': i,
        'params': params,
        'model': rnn,
        'train_loss': t_history,
        'val_loss': v_history,
        'final_val_loss': v_history[-1]
    }

def parallel_grid_search(model, n_iter, config, x_train, y_train, x_val, y_val, seed=42, n_jobs=-1):
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

    model_fn_map = {
        'TDNN': tdnn_single_experiment,
        'RNN': rnn_single_experiment,
    }

    if model in model_fn_map:
        results = Parallel(n_jobs=n_jobs)(
            delayed(model_fn_map[model])(i, config, x_train, y_train, x_val, y_val, seed)
            for i in tqdm(range(n_iter), desc=f"{model} Grid Search")
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    best_result = min(results, key=lambda x: x['final_val_loss'])
    print(f"Best validation loss: {best_result['final_val_loss']}")
    
    return best_result['model'], best_result['train_loss'], best_result['val_loss'], best_result['params']

