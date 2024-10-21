import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
}

plt.rcParams.update(pgf_with_latex)

from src.ltp_system.utils import savefig

output_labels_latex = [r'O$_2$(X)', r'O$_2$(a$^1\Delta_g$)', r'O$_2$(b$^1\Sigma^+_g$)', r'O$_2$(Hz)', 
r'O$_2$(+,X)', r'O($^3$P)', r'O($^1$D)', r'O(+,gnd)', r'O(-,gnd)', r'O$_3$(X)', r'O$_3$(exc)', r'T$_g$', 
r'T$_{nw}$', r'Red$_{E}$', r'v$_{d}$', r'E$_{mean}$', r'n$_{e}$'
]

def Figure_4a(config_plotting, nn_mape_dict, pinn_mape_dict):

    bar_plot_palette = config_plotting['barplot_palette']
    mape_nn = nn_mape_dict['model']['mape']
    mape_pinn = pinn_mape_dict['model']['mape']
    mape_nn_proj = nn_mape_dict['P_I_ne']['mape']
    mape_pinn_proj = pinn_mape_dict['P_I_ne']['mape']
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
    plt.bar(br2, mape_nn_proj,  color=bar_plot_palette[1], width=barWidth, edgecolor='black', label='NN Projection', linewidth=edge_linewidth)
    plt.bar(br3, mape_pinn,  color=bar_plot_palette[2], width=barWidth, edgecolor='black', label='PINN', linewidth=edge_linewidth)
    plt.bar(br4, mape_pinn_proj,  color=bar_plot_palette[3], width=barWidth, edgecolor='black', label='PINN Projection', linewidth=edge_linewidth)

    # Adding Xticks
    plt.ylabel('MAPE (\%)', fontweight='bold', fontsize=24)
    plt.xticks([r + barWidth for r in range(num)], output_labels_latex, rotation=45, fontsize=24, fontweight='bold')
    plt.yticks(fontsize=24)
    
    # Add legend
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.30), ncol=4, fontsize=24, frameon=False)

    # Save figures
    output_dir = config_plotting['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_4a")
    savefig(save_path, pad_inches = 0.2)
    


# Bar plot of relative errors of laws with restrictions pairs
def Figure_4d(results, config_model, config_plotting):
    
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

    plt.bar(br1, [results['model']['mape'][i] for i in relevant_features_idx], yerr=[results['model']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[0], width=barWidth, label=f'{model_type}', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br2, [results['P_I_ne']['mape'][i] for i in relevant_features_idx], yerr=[results['P_I_ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[1], width=barWidth, label=f'{model_type} Projection: all contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br3, [results['P']['mape'][i] for i in relevant_features_idx], yerr=[results['P']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[2], width=barWidth, label=f'{model_type} Projection: P contraint', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br4, [results['I']['mape'][i] for i in relevant_features_idx], yerr=[results['I']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[3], width=barWidth, label=f'{model_type} Projection: I contraint', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br5, [results['ne']['mape'][i] for i in relevant_features_idx], yerr=[results['ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[4], width=barWidth, label=f'{model_type} Projection: ne contraint', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br6, [results['P_I']['mape'][i] for i in relevant_features_idx], yerr=[results['P_I']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[5], width=barWidth, label=f'{model_type} Projection: P and I contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br7, [results['P_ne']['mape'][i] for i in relevant_features_idx], yerr=[results['P_ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[6], width=barWidth, label=f'{model_type} Projection: P and ne contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)
    plt.bar(br8, [results['I_ne']['mape'][i] for i in relevant_features_idx], yerr=[results['I_ne']['sem'][i] for i in relevant_features_idx], color=bar_plot_palette[7], width=barWidth, label=f'{model_type} Projection: I and ne contraints', edgecolor='black', linewidth=edge_linewidth, capsize=capsize)

    # Adding Xticks 
    plt.ylabel('MAPE (\%)', fontsize=24) 
    plt.xticks(x_tick_positions, [output_labels_latex[i] for i in relevant_features_idx], rotation=0, fontsize=30)
    plt.yticks(fontsize=24)
    #plt.ylim(0, 15)

    plt.legend(loc='upper center', bbox_to_anchor=(1.45, 1.1), ncol=1, fontsize=24, frameon=False)

    # Save figures
    output_dir = config_plotting['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_4d")
    savefig(save_path, pad_inches=0.2)



def Figure_4b(config_plotting, laws_dict):

    # Initialize empty lists to hold all the MAPE and SEM values
    all_mapes, all_sems, all_abs_errors = [], [], []

    # Loop through the model types and extract corresponding values
    for model_type in ["loki","nn_model", "pinn_model", "nn_model_proj", "pinn_model_proj"]:
        mapes = [laws_dict[model_type][metric] for metric in ["p_mape", "i_mape", "ne_mape"]]
        sems = [laws_dict[model_type][sem_metric] for sem_metric in ["p_sem", "i_sem", "ne_sem"]]
        abs_errors = [laws_dict[model_type][sem_metric] for sem_metric in ["p_abs_err", "i_abs_err", "ne_abs_err"]]

        all_mapes.append(mapes)
        all_sems.append(sems)
        all_abs_errors.append(abs_errors)

    # Set width of bar 
    width = 0.2
    models = ["LoKI","NN", "PINN", "NN\nProjection", "PINN\nProjection"]

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
        plt.bar(x + i * width, [d[i] for d in all_mapes], width, 
                yerr=[e[i] for e in all_sems], label=label, 
                edgecolor='black', color=bar_plot_palette[i], capsize=capsize, linewidth=edge_linewidth)

    # Customize X and Y axis labels and ticks
    plt.ylabel('MAPE (\%)', fontsize=24)
    plt.xticks([r + width for r in range(len(models))], models, fontsize=24)
    plt.yticks(fontsize=24)
    plt.yscale('log')  # Logarithmic scale
    y_max = 10**4
    y_min = 10**-16
    plt.ylim(y_min, y_max)
    plt.yticks([y_max, 10**0, 10**-4, 10**-8, 10**-12, y_min])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))

    plt.legend(loc='upper center', fontsize=24, frameon=False, bbox_to_anchor=(0.43, 1.35), ncol=3)

    # Save figures
    output_dir = config_plotting['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_4b")
    savefig(save_path, pad_inches = 0.2)


#
def scientific_notation(x, pos):
    return f'{x:.0e}'

