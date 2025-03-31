# Data preparation for lab 2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pylab as plt
import os

def read_data(path = "data/solar_data.csv"):
    """
    read the time series from the path
    
    :param path: the path to the dataset
    
    :return: a numpy array containing the time series
    """
    df = pd.read_csv(path, dtype=float, header=None)
    x = df.to_numpy()
    
    return x

def firing_rate_model(rule, u, n_epochs, lr, delta, alpha=0, theta=0, seed=42):
    """
    Basic Hebbian Learning firing rate model.
    
    :param rule: update rule
    :param u: input data
    :param n_epochs: number of epochs
    :param lr: learning rate
    :param delta: stopping criterion value
    :param alpha: Oja's learning rule parameter
    
    :return w: weight vector
    :return history: evolution of weight vector trough epochs
    :convergence: number of epochs to converge
    """
    np.random.seed(seed)
    w = np.random.uniform(-1, 1, size=2)
    w_old = w.copy()
    
    history = []
    convergence = n_epochs
    
    for epoch in range(1,n_epochs+1):
        X = shuffle(u.T)
        for x in X:
            v = w @ x 
            if rule == "hebb":
                w = w + lr * (v * x)
            elif rule == "oja":
                w = w + lr * ((v * x) - alpha * (v**2) * w)
            elif rule == "sub-norm":
                nu = u.shape[0]
                n = np.ones(nu)
                w = w + lr * (v * x - (v* (n @ x) * n)/nu)
            elif rule == "BCM":
                w = w + lr * (v * x * (v - theta))
                theta = theta + lr * 10 * (v**2 - theta)
            elif rule == "cov":
                C = np.cov(u)
                #w = w + lr * (v * (u - theta))
                w = w + lr * (C @ w)  
            
                
        if np.linalg.norm(w - w_old) < delta:
            print(f"Convergence reach at epoch {epoch}")
            history.append(w.copy())
            convergence = epoch
            break
        
        w_old = w
        history.append(w.copy())
        #if epoch % 100 == 0:
        #    print(f"Running epoch ", epoch)

    return w, np.array(history), convergence

def plot_1(u, w, p_evec, rule, dataset):
    """
    plot number 1: plots input data u, weight vector w and principal eigenvector p_evec
        
    :param u: input data
    :param w: weight vector
    :param p_evec: principal eigenvector
    :param rule: update rule used
    """
    filepath = "outputs/"+dataset+"/"+rule+"/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath += "plot_1.png"
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u[0], u[1], "o", markersize=3, label="Data points")
    plt.xlabel("u[0]")
    plt.ylabel("u[1]")
    plt.title(rule.replace('_', ' ') + " weights and principal eigenvector plot")
    
    if dataset == "dataset_2":
        ax.quiver(0.1, 0.2, w[0], w[1], color="green", width=0.003, label="Weight vector (w)")
        ax.quiver(0.1, 0.2, p_evec[0], p_evec[1], color="red", width=0.003, label="Principal eigenvector")
    else:
        ax.quiver(0, 0, w[0], w[1], color="green", width=0.002, label="Weight vector (w)")
        ax.quiver(0, 0, p_evec[0], p_evec[1], color="red", width=0.002, label="Principal eigenvector")
    
    ax.legend()
    
    if os.path.isfile(filepath):
        os.remove(filepath)
    plt.savefig(filepath)

    plt.show()
    
def plot_2(tspan, w_history, rule, dataset):
    """
    plot number 2: plots the evolution of the weight vector w through epochs
    
    :param w_history: history of the weight vector through epochs
    :param tspan: time span
    :param rule: update rule used
    """
    filepath = "outputs/"+dataset+"/"+rule+"/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath += "plot_2.png"
    
    norm_hist = np.zeros(w_history.shape[0])
    for i, w in enumerate(w_history):
        norm_hist[i] = np.linalg.norm(w)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Evolution of Weight Vector - {rule.replace('_', ' ')}", fontsize=16) 

    # First plot: w[0] over time
    axs[0].plot(tspan, w_history.T[0])
    axs[0].set_title("w[0] over time")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("w[0]")

    # Second plot: w[1] over time
    axs[1].plot(tspan, w_history.T[1])
    axs[1].set_title("w[1] over time")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("w[1]")

    # Third plot: Norm of w over time
    axs[2].plot(tspan, norm_hist)
    axs[2].set_title("Norm of W over time")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Norm of W")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  
    
    if os.path.isfile(filepath):
        os.remove(filepath)
    plt.savefig(filepath)
    plt.show()

def save_weights(weights, rule, dataset):
    filepath = "outputs/"+dataset+"/"+rule+"/"
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath += "weights.npz"
    np.savez(filepath, *weights)
