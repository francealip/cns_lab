import random
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from esn import Esn
import torch

def esn_single_experiment(n_iter, i, config, x_train, y_train, x_val, y_val, seed=42):
    """
    Run a single experiment with a specific configuration with RNN.
    
    :param n_iter: Number of iterations for the experiment.
    :param i: Experiment index.
    :param config: Configurations for the RNN model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param seed: Random seed for reproducibility.
    """
    random.seed(seed * i)

    params = {k: random.choice(v) for k, v in config.items()}
    
    train_loss = []
    val_loss = []
    
    for k in range(n_iter):
        
        esn = Esn(
            in_dim=x_train.shape[1],
            out_dim=y_train.shape[1],
            hidden_dim=params['hidden_dim'],
            rho=params['rho'],
            keep_prob=params['keep_prob'],
            alpha=params['alpha'],
            omega_in=params['omega_in'],
            omega_bias=params['omega_bias'],
            scaling_type=params['scaling_type'],
            washout = params['washout'],
        )

        esn.fit(
            x_train = x_train,
            y_train = y_train,
            lambd=params['lambd'],
        )
        
        t_loss = esn.loss(esn.forward(x_train), y_train)
        v_loss = esn.loss(esn.forward(x_val), y_val)

        train_loss.append(t_loss.item())
        val_loss.append(v_loss.item())

    return {
        'iteration': i,
        'params': params,
        'model': esn,
        'train_loss': (np.mean(train_loss), np.std(train_loss)),
        'val_loss': (np.mean(val_loss), np.std(val_loss)),
    }

def parallel_grid_search(random_iter, n_iter, config, x_train, y_train, x_val, y_val, seed=42, n_jobs=-1):
    """
    Perform parallel grid search for TDNN hyperparameters.
    
    :random_iter: Number of random iterations for the grid search.
    :param n_iter: Number of iterations for each experiment.
    :param config: Configurations for the TDNN model.
    :param x_train: Training data.
    :param y_train: Training labels.
    :param x_val: Validation data.
    :param y_val: Validation labels.
    :param seed: Random seed for reproducibility.
    :param n_jobs: Number of parallel jobs to run.
    
    :return: Best model, training loss mean and std, validation loss mean and std, and best parameters.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(esn_single_experiment)(n_iter, i, config, x_train, y_train, x_val, y_val, seed)
        for i in tqdm(range(random_iter), desc=f"ESN Grid Search")
    )

    best_result = min(results, key=lambda x: x['val_loss'][0])
    print(f"Best validation loss: {best_result['val_loss']}")
    
    return best_result['model'], best_result['train_loss'], best_result['val_loss'], best_result['params']
