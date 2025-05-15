import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob


def read_data(path):
    csv_files = glob.glob(path + "*.csv")
    num_patterns = len(csv_files)
    file_csv = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    X = torch.empty(num_patterns, 1024)

    for i,file in enumerate(file_csv):
        if file.lower().endswith(".csv"):
            df = pd.read_csv(path+ file, dtype=float, header=None)
            x = torch.tensor(df.values).reshape(1, 1024)
            X[i] = x      # because of a DS_store file inside
            
    return X


def distortion(p, d):
    """
    Apply a random distortion to d% values on an input vector p
    
    :param p: the input vector to be distorted
    :param d: a scalar value in [0,1] that considers the percentage of values to be randomly flipped
    
    :return: the distorted array
    """
    n = p.size(0)
    k = int(n * d)       #proportion of values to be permutated
    
    torch.manual_seed(42)
    sample = torch.randperm(n)[:k].unsqueeze(0) 
    mask = torch.ones(1, n, dtype=torch.float).scatter_(1, sample, -1)
        
    return p * mask


def plot_images(original, noisy, reconstruction, noise, folder):
    """
    Display side by side: original image, noisy image, and reconstructed image.

    :param original: original image
    :param noisy: noisy image
    :param reconstruction: image obtained from the Hopfield network
    :param noise: intensity of the applied noise
    """
    original = original.reshape(32, 32)
    noisy = noisy.reshape(32, 32)
    reconstruction = reconstruction.reshape(32, 32)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle(f'Image Comparison\n', fontsize=16)

    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original Image')

    axs[1].imshow(noisy, cmap='gray')
    axs[1].set_title(f'Noisy Image (level: {noise})')

    axs[2].imshow(reconstruction, cmap='gray')
    axs[2].set_title('Reconstructed Image')

    plt.tight_layout()
    plt.savefig(os.path.join(f'plots/{folder}/', f'reconstruction_{noise}_noise.png'))
    plt.show()


def plot_energy_overlap(energy, overlap, noise, folder):
    """
    Display side by side: energy  across iterations and overlap across iterations.

    :param energy: energy across iterations
    :param overlap: overlap across iterations
    :param noise: intensity of the applied noise
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle(f'Energy and Overlap history with {noise} noise \n', fontsize=16)
    
    axs[0].plot(energy, color='gold')
    axs[0].set_title('Energy across updates')
    axs[0].set_ylabel('Energy')
    axs[0].set_xlabel('Updates')

    axs[1].plot(overlap, color='purple')
    axs[1].set_title('Overlap across updates')
    axs[1].set_ylabel('overlap')
    axs[1].set_xlabel('Updates')
    
    plt.tight_layout()
    plt.savefig(os.path.join(f'plots/{folder}/', f'energy_overlap_{noise}_noise.png'))
    plt.show()
