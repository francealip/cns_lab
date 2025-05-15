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


def import_parameters(yaml2, yaml4="tdnn4.yaml"):
    """
    Import parameters from YAML configuration files for 2 layer and 4 layer TDNN.
    
    :param yaml2: Path to the YAML file for 2 layer TDNN.
    :param yaml4: Path to the YAML file for 4 layer TDNN.
    
    :return: Tuple of configurations for 2 layer and 4 layer TDNN.
    """
    # Load the YAML configuration files for 2 layer and 4 layer TDNN
    with open('grid_param/'+yaml2, 'r') as f:
        config1 = yaml.safe_load(f)

    with open('grid_param/'+yaml4, 'r') as f:
        config2 = yaml.safe_load(f)
        
    return config1, config2


def plot_histories(model, t_history, v_history, val_set="Validation"):
    """
    Plot the training and validation loss histories.
    
    :param t_history: Training loss history.
    :param v_history: Validation loss history.
    :param val_set: Name of the validation set, default "Validation".
    """
    plt.figure(figsize=(10, 5))
    plt.plot(t_history, label='Training History')
    plt.plot(v_history, label=val_set + ' History')
    plt.title('Training and ' + val_set +' MSE Histories')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{model}/training_{val_set}_history.png')
    plt.show()
    
    
def plot_predictions(y_true, y_pred, title="Predicted vs True Values", num_instances=250, model="tdnn"):
    """
    Plot the predicted values against the true values over time.
    """
    plt.figure(figsize=(20, 5))
    plt.plot(y_true.flatten().detach().numpy()[:num_instances], label="True Values", alpha=0.7)
    plt.plot(y_pred.flatten().detach().numpy()[:num_instances], label="Predicted Values", alpha=0.7, linestyle=':')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    set = title.split(" ")[0].lower()
    plt.savefig(f'results/{model}/{set}_set_predictions_over_time.png')
    plt.show()

