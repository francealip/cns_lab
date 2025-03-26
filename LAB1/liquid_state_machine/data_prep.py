# Data preparation for the solar panel dataset

import pandas as pd
import numpy as np

def read_data(path = "data/solar_data.csv"):
    """
    read the time series from the path
    
    :param path: the path to the dataset
    
    :return: a numpy array containing the time series
    """
    df = pd.read_csv(path, dtype=float, header=None)
    x = df.to_numpy()
    x = x.squeeze()
    
    return x


def train_test_split(x, train_p=0.65, val_p=0.175, test_p=0.175):
    """
    split the time series into training, validation and test sets

    :param x: the time series
    :param train_p: proportion of the data to be used for training
    :param val_p: proportion of the data to be used for validation
    :param test_p: proportion of the data to be used for testing
    
    :return: training_x, training_y, validation_x, validation_y, test_x, test_y
    
    """
    assert((train_p + val_p + test_p) == 1)
    
    n = len(x) - 1  # One less because we shift for next-step prediction

    tr_end = int(n * train_p)
    val_end = tr_end + int(n * val_p)

    train_x, train_y = x[:tr_end], x[1:tr_end+1]
    val_x, val_y = x[tr_end:val_end], x[tr_end+1:val_end+1]
    test_x, test_y = x[val_end:-1], x[val_end+1:]  # we don't have the target for the last point
    
    return train_x, train_y, val_x, val_y, test_x, test_y
    
    
    