# Data preparation for lab 2
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pylab as plt

def read_data(path = "data/solar_data.csv"):
    """
    read the time series from the path
    
    :param path: the path to the dataset
    
    :return: a numpy array containing the time series
    """
    df = pd.read_csv(path, dtype=float, header=None)
    x = df.to_numpy()
    
    return x

def firing_rate_model(rule, u, n_epochs, lr, delta, alpha=0, seed=42):
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
                n = np.ones(u.shape[0])
                nu = u.shape[1]
                
                w = w + lr * ((v * x) - (((v*(n@x))*n)/nu))

        if np.linalg.norm(w - w_old) < delta:
            print(f"Convergence reach at epoch {epoch}")
            history.append(w.copy())
            convergence = epoch
            break
        
        w_old = w
        history.append(w.copy())
        if epoch % 100 == 0:
            print(f"Running epoch ", epoch)

    return w, np.array(history), convergence

def plot_1(u, w, p_evec):
    """
    plot number 1: plots input data u, weight vector w and principal eigenvector p_evec
        
    :param u: input data
    :param w: weight vector
    :param p_evec: principal eigenvector
    """
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(u[0], u[1], "o", markersize=3, label="Data points")
    plt.xlabel("u[0]")
    plt.ylabel("u[1]")
    plt.title("lab2 dataset")
    ax.quiver(0, 0, w[0], w[1], color="green", width=0.002, label="Weight vector (w)")
    ax.quiver(0, 0, p_evec[0], p_evec[1], color="red", width=0.002, label="Principal eigenvector")
    ax.legend()

    plt.show()
    
def plot_2_0(tspan, w_history):
    """
    plot number 2: plots the evolution of the weight vector through epochs component by component
    
    :param w_history: history of the weight vector trough epochs
    :param tspan: time span
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # First plot
    axs[0].plot(tspan, w_history.T[0])
    axs[0].set_title("w[0] over time")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("w[0]")

    # Second plot
    axs[1].plot(tspan, w_history.T[1])
    axs[1].set_title("w[1] over time")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("w[1]")

    plt.show()

def plot_2_1(tspan, w_history, epochs):
    """
    plot number 2: plots the evolution of the weight vector through epochs
    
    :param w_history: history of the weight vector trough epochs
    :param tspan: time span
    :param epochs: number of epochs
    """
    
    norm_hist = np.zeros(epochs)
    for i,w in enumerate(w_history):
        norm_hist[i] = np.linalg.norm(w)

    plt.figure(figsize=(6,4))
    plt.plot(tspan, norm_hist)
    plt.title("Norm of W over time")
    
    plt.show()