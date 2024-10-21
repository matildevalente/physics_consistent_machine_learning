import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
from typing import Dict, Any
import matplotlib.pyplot as plt
from scipy import stats

from src.ltp_system.utils import figsize, savefig
def plot_output_gaussians(config, *diff_arrays):
    len_outputs = len(diff_arrays)
    variable_names = config['dataset_generation']['output_features']

    # Create histogram plot
    plt.clf()
    fig, axs = plt.subplots(3, 6, figsize=(22, 12))
    for i, (ax, diff, var_name) in enumerate(zip(axs.flat, diff_arrays, variable_names)):
        ax.hist(diff, bins=30, edgecolor='black')
        ax.set_title(f"NN residuals: {var_name} (mean = {np.mean(diff):.2e})")
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    output_dir = config['plotting']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"EDA/normality/Extra_Figure_1.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()  # Close the figure to free up memory


    # Create Q-Q plot
    fig, axs = plt.subplots(3, 6, figsize=(22, 12))
    for i, (ax, diff, var_name) in enumerate(zip(axs.flat, diff_arrays, variable_names)):
        # Create Q-Q plot
        (osm, osr), _ = stats.probplot(diff, dist="norm", plot=ax)
        
        # Customize the plot
        ax.set_title(f"Q-Q Plot: NN residuals {var_name}")
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')
        
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        
        # Perform Shapiro-Wilk test
        stat, p = stats.shapiro(diff)
        
        # Add test results and other statistics to the plot
        textstr = f'Shapiro-Wilk test:\np-value: {p:.2e}\n'
        textstr += f'W statistic: {stat:.3f}\n'
        textstr += f'Mean: {np.mean(diff):.2e}\n'
        textstr += f'Std Dev: {np.std(diff):.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    # Save Q-Q plot figure    
    save_path = os.path.join(output_dir, f"EDA/normality/Extra_Figure_1_qq_plot.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.2)
    plt.close()  # Close the figure to free up memory