# Utility functions for the project

import matplotlib.pyplot as plt
import yaml


def plot_time_series(time, data, title, xlabel, ylabel, limit=200):
    """
    Plot first "limit" points of time series data.
    
    :param time: Time data.
    :param data: Time series values to be plotted.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param limit: Number of points to plot.
    """
    plt.figure(figsize=(20, 5))
    plt.plot(time[:limit], data[:limit])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()
   
    
def split_data(X, Y, train_size, val_size):
    """
    Split the data into training, validation, and test sets.
    
    :param X: Input data.
    :param Y: Target data.
    :param train_size: Size of the training set.
    :param val_size: Size of the validation set.
    
    :return: Tuple of training, validation, and test sets.
    """
    x_train, x_val, x_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = Y[:train_size], Y[train_size:train_size + val_size], Y[train_size + val_size:]
    
    return x_train, x_val, x_test, y_train, y_val, y_test


def import_parameters(file):
    """
    Import parameters from YAML configuration files for 2 layer and 4 layer TDNN.
    
    :param file: Path to the YAML file for ESN.
    
    :return: Tuple of configurations for 2 layer and 4 layer TDNN.
    """
    # Load the YAML configuration files for 2 layer and 4 layer TDNN
    with open('grid_param/'+file, 'r') as f:
        config = yaml.safe_load(f)
        
    return config
    
    
def plot_predictions(y_true, y_pred, title="Predicted vs True Values", start = 100, end=250):
    plt.figure(figsize=(20, 5))
    plt.plot(y_true.flatten().detach().numpy()[start:end], label="True Values", alpha=0.7)
    plt.plot(y_pred.flatten().detach().numpy()[start:end], label="Predicted Values", alpha=0.7, linestyle=':')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    name = title.split(" ")[0].lower() + "_plot"
    plt.savefig("results/"+name + ".png")
    
    plt.show()
