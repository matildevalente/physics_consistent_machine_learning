import os
import torch
import numpy as np
import matplotlib as mpl
from typing import List, Dict
import matplotlib.pyplot as plt
from src.ltp_system.utils import savefig
from matplotlib.ticker import FuncFormatter


pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
}

plt.rcParams.update(pgf_with_latex)

output_labels = [r'O$_2$(X)', r'O$_2$(a$^1\Delta_g$)', r'O$_2$(b$^1\Sigma_g^+$)', r'O$_2$(Hz)', r'O$_2^+$', r'O($^3P$)', r'O($^1$D)', r'O$^+$', r'O$^-$', r'O$_3$', r'O$_3^*$', r'$T_g$', r'T$_{nw}$', r'$E/N$', r'$v_d$', r'T$_{e}$', r'$n_e$']

# Code for plotting barplot of the MAPE for each of the output features
def Figure_4a(config_plotting, nn_mape_dict, pinn_mape_dict, error_type):

    bar_plot_palette = config_plotting['barplot_palette']
    mape_nn = nn_mape_dict['model'][error_type]
    mape_pinn = pinn_mape_dict['model'][error_type]
    mape_nn_proj = nn_mape_dict['P_I_ne'][error_type]
    mape_pinn_proj = pinn_mape_dict['P_I_ne'][error_type]
    num = len(mape_nn)

    # Plot configuration
    barWidth = 0.2
    plt.clf()
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(16, 4))
    edge_linewidth, capsize = 0.2, 4
    
    # Set position of bar on X axis
    br1 = np.arange(num)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    plt.bar(br1, mape_nn,  color=bar_plot_palette[0], width=barWidth, edgecolor='black', label='NN', linewidth=edge_linewidth)
    plt.bar(br2, mape_nn_proj,  color=bar_plot_palette[1], width=barWidth, edgecolor='black', label='NN projection', linewidth=edge_linewidth)
    plt.bar(br3, mape_pinn,  color=bar_plot_palette[2], width=barWidth, edgecolor='black', label='PINN', linewidth=edge_linewidth)
    plt.bar(br4, mape_pinn_proj,  color=bar_plot_palette[3], width=barWidth, edgecolor='black', label='PINN projection', linewidth=edge_linewidth)

    # Adding Xticks
    if error_type == 'mape':
        plt.ylabel('MAPE (\%)', fontweight='bold', fontsize=24)
    elif error_type == 'rmse':
        plt.ylabel('RMSE', fontweight='bold', fontsize=24, labelpad=10)
    plt.xticks([r + barWidth for r in range(num)], output_labels, rotation=45, fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24)
    
    if error_type == 'rmse':
        # Set y-axis to log scale and range
        plt.yscale('log')
        plt.ylim(5e-3, 50e-3)
        plt.yticks([5e-3, 10e-3, 20e-3, 40e-3])
        plt.gca().set_yticklabels(['5', '10', '20', '40'])
        plt.minorticks_off()
        # Get the maximum y-axis value
        y_max = plt.gca().get_ylim()[1]
        # Add scaling text
        plt.text(0, y_max * 1.12, r'($\times10^{-3}$)', transform=plt.gca().get_yaxis_transform(), fontsize=20)
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.335), ncol=4, fontsize=24, frameon=False)
    
    # Save figures
    output_dir = config_plotting['output_dir'] + "Figures_4/Figure_4a/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_4a_{error_type}")
    savefig(save_path, pad_inches = 0.2)
    
# Code for plotting relative errors of laws with restrictions pairs
def Figure_4d(results, config_model, config_plotting, error_type):
    
    # Determine the model type being analyzed
    model_type = "NN" if all(param == 0 for param in config_model['lambda_physics']) else "PINN"

    # Select the outputs relevant for the plot
    relevant_features_idx = [0, 4, 16]
    
    # Set width of bar 
    barWidth = 0.11
    
    plt.clf()
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 4))

    # Set position of bar on X axis 
    br1 = np.arange(len(relevant_features_idx)) 
    br2 = [x + barWidth for x in br1] 
    br3 = [x + barWidth for x in br2] 
    br4 = [x + barWidth for x in br3] 
    br5 = [x + barWidth for x in br4] 
    br6 = [x + barWidth for x in br5] 
    br7 = [x + barWidth for x in br6] 
    br8 = [x + barWidth for x in br7] 

    # Make the plot with error bars
    edge_linewidth, capsize = 0.5, 10
    bar_plot_palette = config_plotting['barplot_palette']
    x_tick_positions = [(x + barWidth * 3.5) for x in br1]  # Center between the 4th and 5th bar

    plt.bar(br1, [results['model'][error_type][i] for i in relevant_features_idx], yerr=[results['model']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[0], width=barWidth, label=f'{model_type}', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br2, [results['P_I_ne'][error_type][i] for i in relevant_features_idx], yerr=[results['P_I_ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[1], width=barWidth, label=f'{model_type} projection: all contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br3, [results['P'][error_type][i] for i in relevant_features_idx], yerr=[results['P']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[2], width=barWidth, label=f'{model_type} projection: P contraint', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br4, [results['I'][error_type][i] for i in relevant_features_idx], yerr=[results['I']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[3], width=barWidth, label=f'{model_type} projection: I contraint', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br5, [results['ne'][error_type][i] for i in relevant_features_idx], yerr=[results['ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[4], width=barWidth, label=f'{model_type} projection: ne contraint', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br6, [results['P_I'][error_type][i] for i in relevant_features_idx], yerr=[results['P_I']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[5], width=barWidth, label=f'{model_type} projection: P and I contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br7, [results['P_ne'][error_type][i] for i in relevant_features_idx], yerr=[results['P_ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[6], width=barWidth, label=f'{model_type} projection: P and ne contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br8, [results['I_ne'][error_type][i] for i in relevant_features_idx], yerr=[results['I_ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[7], width=barWidth, label=f'{model_type} projection: I and ne contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)

    # Adding Xticks 
    if error_type == 'mape':
        plt.ylabel('MAPE (\%)', fontweight='bold', fontsize=24)
    elif error_type == 'rmse':
        plt.ylabel('RMSE', fontweight='bold', fontsize=24, labelpad=10)
        # Set y-axis to log scale and range
        plt.ylim(5e-3, 26e-3)
        plt.yticks([5e-3,10e-3,15e-3,20e-3, 25e-3])
        plt.gca().set_yticklabels(['5', '10', '15', '20', '25'])
        plt.minorticks_off()
        # Get the maximum y-axis value
        y_max = plt.gca().get_ylim()[1]
        plt.text(0, y_max * 1.05, r'($\times10^{-3}$)', transform=plt.gca().get_yaxis_transform(), fontsize=20)
    plt.xticks(x_tick_positions, [output_labels[i] for i in relevant_features_idx], rotation=0, fontsize=30)
    plt.yticks(fontsize=24)
    #plt.ylim(0, 15)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), ncol=1, fontsize=24, frameon=False)

    # Save figures
    output_dir = config_plotting['output_dir'] + "Figures_4/Figure_4d/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_4d_{error_type}")
    savefig(save_path, pad_inches=0.2)

# Code for plotting the error in compliance with physical laws for each of the models
def Figure_4b(config_plotting, laws_dict, error_type):

    # Initialize empty lists to hold all the MAPE and SEM values
    all_errors, all_sems, all_abs_errors = [], [], []

    # Loop through the model types and extract corresponding values
    for model_type in ["nn_model", "pinn_model", "nn_model_proj", "pinn_model_proj"]:
        errors = [laws_dict[model_type][metric] for metric in [f"p_{error_type}", f"i_{error_type}", f"ne_{error_type}"]]
        sems = [laws_dict[model_type][sem_metric] for sem_metric in ["p_sem", "i_sem", "ne_sem"]]
        #abs_errors = [laws_dict[model_type][sem_metric] for sem_metric in ["p_abs_err", "i_abs_err", "ne_abs_err"]]

        all_errors.append(errors)
        all_sems.append(sems)
        #all_abs_errors.append(abs_errors)

    # Set width of bar 
    width = 0.2
    models = ["NN", "PINN", "NN\nprojection", "PINN\nprojection"]

    x = np.arange(len(models))
    palette = config_plotting['barplot_palette']
    bar_plot_palette = [palette[2], palette[3], palette[4]]
    
    plt.clf()
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 4))
    # Plot bars for each category with error bars
    edge_linewidth, capsize = 0.2, 4
    for i, label in enumerate(["Ideal Gas Law", "I Law", "Quasi Neutrality Law"]):
        plt.bar(x + i * width, [d[i] for d in all_errors], width, 
                yerr=[e[i] for e in all_sems], label=label, 
                edgecolor='black', color=bar_plot_palette[i], capsize=capsize, linewidth=edge_linewidth)

    # Customize X and Y axis labels and ticks
    if error_type == 'mape':
        plt.ylabel('MAPE (\%)', fontsize=24, fontweight='bold')
    elif error_type == 'rmse':
        plt.ylabel('RMSE', fontsize=26, fontweight='bold')
    plt.xticks([r + width for r in range(len(models))], models, fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24)
    plt.yscale('log')  # Logarithmic scale
    y_max = 10**4
    y_min = 10**-18
    plt.ylim(y_min, y_max)
    plt.yticks([y_max, 10**0, 10**-4, 10**-8, 10**-12, 10**-16])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))

    plt.legend(loc='upper center', fontsize=24, frameon=False, bbox_to_anchor=(0.43, 1.35), ncol=3)

    # Save figures
    output_dir = config_plotting['output_dir'] + "Figures_4/Figure_4b/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_4b_{error_type}")
    savefig(save_path, pad_inches = 0.2)

# Guarantee the values are in scientific notation
def scientific_notation(x, pos):
    return f'{x:.0e}'

