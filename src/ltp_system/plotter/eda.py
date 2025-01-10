import os
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
from typing import Dict, Any
import matplotlib.pyplot as plt


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

output_labels_latex = [r'O$_2$(X)', r'O$_2$(a$^1\Delta_g$)', r'O$_2$(b$^1\Sigma_g^+$)', r'O$_2$(Hz)', r'O$_2^+$', r'O($^3P$)', r'O($^1$D)', r'O$^+$', r'O$^-$', r'O$_3$', r'O$_3^*$', r'$T_g$', r'T$_{nw}$', r'$E/N$', r'$v_d$', r'T$_{e}$', r'$n_e$']



from src.ltp_system.utils import figsize, savefig

# Histogram of the Normalized Dataset
def features_histogram_(config, X_train_norm, y_train_norm, X_test_norm, y_test_norm):

    palette = config['plotting']['palette']
    all_features = config['dataset_generation']['input_features'] + config['dataset_generation']['output_features']

    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    
    for idx, feature in enumerate(all_features):

        plt.figure(figsize=(10, 7))
        if(idx < 3):
            plt.hist(X_train_norm[:, idx], bins=30, color=palette[0], alpha=0.75, label=feature + " (Training Set)", edgecolor='black')
            plt.hist(X_test_norm[:, idx], bins=30, color=palette[1], alpha=0.75, label=feature + " (Test Set)", edgecolor='black')
        else:
            opt_idx = idx - 3
            plt.hist(y_train_norm[:, opt_idx], bins=40, color=palette[2], alpha=0.75, label= feature + " (Training Set)", edgecolor='black')
            plt.hist(y_test_norm[:, opt_idx], bins=40, color=palette[3], alpha=0.75, label= feature + " (Test Set)", edgecolor='black')

        plt.xlabel(feature, fontsize=16, fontweight='bold')
        plt.ylabel('Frequency', fontsize=16, fontweight='bold')
        plt.title(f'{feature} Histogram', fontsize=18, fontweight='bold')

        plt.legend(fontsize='large', loc='upper right')
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')

        plt.xlim(-1, 1)

        # Ensure all ticks on the x- and y- axes are labeled
        plt.tick_params(axis='x', which='both', direction='in', pad=10)
        plt.tick_params(axis='y', which='both', direction='in', pad=10)

        # Bold bounding box
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)

        # Save figure
        output_dir = config['plotting']['output_dir'] + "EDA/histograms/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{feature}")
        savefig(save_path, pad_inches = 0.2)

# Pie Chart with the Species Density
def densities_piechart_(config, y_data):
    # Column indices for species densities
    species_indices = range(11)
    palette = config['plotting']['palette']

    # Extract species densities
    species_densities = y_data[:, species_indices]

    # Calculate mean density for each species
    mean_densities = np.mean(species_densities, axis=0)

    # Identify the indices of the three most dominant species
    dominant_indices = np.argsort(mean_densities)[-3:]

    # Sum the densities of the remaining species into "other"
    other_density = np.sum(mean_densities[np.setdiff1d(np.arange(len(mean_densities)), dominant_indices)])

    # Prepare data for the pie chart
    dominant_densities = mean_densities[dominant_indices]
    dominant_labels = [output_labels_latex[i] for i in dominant_indices]
    dominant_densities = np.append(dominant_densities, other_density)
    dominant_labels.append('Other')
    
    # Colors from the tab10 color map
    colors = [palette[i] for i in dominant_indices] + ['#7f7f7f']

    # Plotting the pie chart
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(7, 5))
    wedges, texts = plt.pie(dominant_densities, colors=colors, startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.7))

    # Add percentages outside the pie chart
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        plt.annotate(f'{dominant_densities[i]/sum(dominant_densities)*100:.1f}\\%',
                    xy=(x, y), xytext=(1.4*np.sign(x), 1.5*y),
                    horizontalalignment=horizontalalignment, fontsize=30, fontweight='bold',
                    arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle, linewidth=2)) 


    # Add a legend
    legend = plt.legend(wedges, dominant_labels, title="Species", loc="lower left", bbox_to_anchor=(0.8, 0.5), fontsize=30)
    plt.setp(legend.get_title(), fontsize=35)
    plt.axis('equal')  

    # Save figure
    output_dir = config['plotting']['output_dir'] + "Figures_4/Figure_4c/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"pie_plot")
    savefig(save_path, pad_inches = 0.2)


# study training data boundaries and plot histograms
def apply_eda(config, dataset, y_data):
  
  for i in range(len(dataset.X_train_norm[0,:])):
    min_value = torch.min(torch.tensor(dataset.X_train_norm[i,:])).item()
    max_value = torch.max(torch.tensor(dataset.X_train_norm[i,:])).item()
    print("input ", dataset.input_features[i], "\n [ ", min_value, "  ,  ", max_value , " ] ")
  
  for i in range(len(dataset.y_train_norm[0,:])):
    min_value = torch.min(torch.tensor(dataset.y_train_norm[i,:])).item()
    max_value = torch.max(torch.tensor(dataset.y_train_norm[i,:])).item()
    print("output ", dataset.output_features[i],  "\n [ ", min_value, "  ,  ", max_value, " ] ")

  features_histogram_(config, dataset.X_train_norm, dataset.y_train_norm, dataset.X_test_norm, dataset.y_test_norm)
  densities_piechart_(config, y_data)

  # Print minimum and maximum values of each column
  for column in (dataset.df_train).columns:
    min_value = dataset.df_train[column].min()
    max_value = dataset.df_train[column].max()
    print(f"Column '{column}': Min Value: {min_value}, Max Value: {max_value}")
