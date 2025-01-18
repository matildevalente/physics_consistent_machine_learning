import os
import time
import math
import torch 
import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt
from itertools import zip_longest
from sciplotlib import style as spstyle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter

from src.ltp_system.utils import savefig




# Training Loss Curve: train vs. val loss_curve_nn(config, nn_losses_dict)
def loss_curves(config_model, config_plotting, losses_dict):

    if config_model['lambda_physics'] == [0,0,0]:
        model_name = "NN"
    else:
        model_name = "PINN"

    plt.clf()
    # Set font to sans-serif
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    # Make plots
    plt.figure(figsize=(7, 5))

    # Use Nature-branded colors for each plot line
    plt.errorbar(
        losses_dict['epoch'], losses_dict['losses_train_mean'], yerr= losses_dict['losses_train_err'],
        label=r'Training loss $\mathcal{L}_{\mathrm{train}}$', markersize=2, color='#1f77b4', capsize=3, elinewidth=2, markeredgewidth=2, linewidth=2 
    )#
    plt.errorbar(
        losses_dict['epoch'], losses_dict['losses_val_mean'], yerr= losses_dict['losses_val_err'],
        label=r'Validation loss $\mathcal{L}_{\mathrm{val}}$', markersize=2, color='#ff7f0e', capsize=3, elinewidth=2, markeredgewidth=2, linewidth=2 
    )

    if(model_name == 'PINN'):
        plt.errorbar(
            losses_dict['epoch'], losses_dict['losses_train_data_mean'], yerr= losses_dict['losses_train_data_err'],
            label=r'Weighted $\mathcal{L}_{\mathrm{data}}$', markersize=2, color='#2ca02c', capsize=3, elinewidth=2, markeredgewidth=2, linewidth=2 
        )
        plt.errorbar(
            losses_dict['epoch'], losses_dict['losses_train_physics_mean'], yerr= losses_dict['losses_train_physics_err'],
            label=r'Weighted $\mathcal{L}_{\mathrm{physics}}$', markersize=2, color='#d62728', capsize=3, elinewidth=2, markeredgewidth=2, linewidth=2 
        )

    plt.yscale('log')
    plt.xlabel('Epochs', fontsize=16, fontweight='bold')
    plt.ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$', fontsize=20, fontweight='bold')  # Use \mathrm instead of \text


    plt.legend(fontsize=16, loc='upper right')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tick_params(axis='both', which='minor', labelsize=16)

    # Ensure all ticks on the x- and y- axes are labeled
    plt.gca().xaxis.set_tick_params(which='both', direction='in', top=True, bottom=True)
    plt.gca().yaxis.set_tick_params(which='both', direction='in', left=True, right=True)


    
    # Bold bounding box
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    # Save figures
    output_dir = config_plotting['output_dir'] + "loss_curves/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name}_loss_curves")
    savefig(save_path, pad_inches = 0.2)

