import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
from typing import Dict, Any
import matplotlib.pyplot as plt
from scipy import stats



pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
}

plt.rcParams.update(pgf_with_latex)


from src.spring_mass_system.utils import figsize, savefig

def plot_output_gaussians(config, x1_diff, v1_diff, x2_diff, v2_diff, label):


    # Create plot using LaTeX for pgf
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].hist(x1_diff, bins=30, color='blue', edgecolor='black')
    axs[0, 0].set_title(f"NN residuals: x_1 (mean = {np.mean(x1_diff):.2e})")
    axs[0, 1].hist(v1_diff, bins=30, color='green', edgecolor='black')
    axs[0, 1].set_title(f"NN residuals: v_1 (mean = {np.mean(v1_diff):.2e})")
    axs[1, 0].hist(x2_diff, bins=30, color='red', edgecolor='black')
    axs[1, 0].set_title(f"NN residuals: x_2 (mean = {np.mean(x2_diff):.2e})")
    axs[1, 1].hist(v2_diff, bins=30, color='purple', edgecolor='black')
    axs[1, 1].set_title(f"NN residuals: v_2 (mean = {np.mean(v2_diff):.2e})")
    for ax in axs.flat:
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    # Save figures
    output_dir = config['plotting']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "extra_figures/Extra_Figure_1_"+label)
    savefig(save_path, pad_inches = 0.2)

    # Create plot using LaTeX for pgf
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    colors = ['blue', 'green', 'red', 'purple']
    data = [x1_diff, v1_diff, x2_diff, v2_diff]
    titles = ['x_1', 'v_1', 'x_2', 'v_2']
    for i, (ax, color, d, title) in enumerate(zip(axs.flat, colors, data, titles)):
        # Create Q-Q plot
        (osm, osr), _ = stats.probplot(d, dist="norm", plot=ax)
        
        # Customize the plot
        ax.set_title(f"Q-Q Plot: NN residuals {title}")
        ax.get_lines()[0].set_markerfacecolor(color)
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')  # Set the line color to red
        
        # Set labels
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        
        # Perform Shapiro-Wilk test
        stat, p = stats.shapiro(d)
        
        # Add test results and other statistics to the plot
        textstr = f'Shapiro-Wilk test:\np-value: {p:.2e}\n'
        textstr += f'W statistic: {stat:.3f}\n'
        textstr += f'Mean: {np.mean(d):.2e}\n'
        textstr += f'Std Dev: {np.std(d):.2e}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    # Save figures
    output_dir = config['plotting']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "extra_figures/Extra_Figure_1_qq_plot"+label)
    savefig(save_path, pad_inches = 0.2)


    