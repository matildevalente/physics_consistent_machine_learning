import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

from src.spring_mass_system.utils import figsize, savefig

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

models_parameters = {
    'NN': {
        'name': 'NN',
        'color': '#0b63a0',
    },
    'PINN': {
        'name': 'PINN',
        'color': '#FFAE0D'  
    },
    'proj_nn': {
        'name': 'NN Projection',
        'color': '#d62728'  
    },
    'proj_pinn': {
        'name': 'PINN Projection',
        'color': '#9A5092'  
    }
}


# Figure 2b
def plot_predicted_trajectory_vs_target(
        config: Dict[str, Any], 
        df_target: pd.DataFrame, 
        df_nn: pd.DataFrame, 
        df_pinn: pd.DataFrame, 
        df_proj_nn: pd.DataFrame, 
        df_proj_pinn: pd.DataFrame
    ):

    variables = ['x1', 'v1', 'x2', 'v2']
    y_labels = ["x$_1$ (m)", "v$_1$ (m/s)", "x$_2$ (m)", "v$_2$ (m/s)"]
    y_labels_units = ["(m)", "(m/s)", "(m)", "(m/s)"]
    legend_colors, legend_labels = [], []

    models_parameters['NN']['pred_trajectory'] = df_nn
    models_parameters['PINN']['pred_trajectory'] = df_pinn
    models_parameters['proj_nn']['pred_trajectory'] = df_proj_nn
    models_parameters['proj_pinn']['pred_trajectory'] = df_proj_pinn
    
    global_mape_max = 0

    for model_key in ['NN', 'PINN', 'proj_nn', 'proj_pinn']:
        df_pred = models_parameters[model_key]['pred_trajectory']
        for i, var in enumerate(variables):
            err_arr = np.abs( df_pred[var] - df_target[var] ) 
            models_parameters[model_key]['abs_err'] = err_arr

    global_mape_max = np.ceil(max(np.max(models_parameters[model_key]['abs_err']) for model_key in ['NN', 'PINN', 'proj_nn', 'proj_pinn']) * 10) / 10

    # Use LaTeX for pgf
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(20, 15))
    legend_colors.append("#2ca02c")
    legend_labels.append("Target")

    # Define list with the names of the models to include in the plot
    models_for_plot = ['NN', 'PINN', 'proj_nn', 'proj_pinn']
    num_models_for_plot = len(models_for_plot)

    if(num_models_for_plot == 3):
        # The 4th column is used as blank space
        gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 0.3, 1], wspace=0.1)  

        # X ranges for the yellow shaded areas of the plots
        zoom_time_ranges = [(6, 8), (5, 7), (5, 7), (5,7)]

    elif(num_models_for_plot == 4):
        # The 4th column is used as blank space
        gs = gridspec.GridSpec(4, 6, width_ratios=[1, 1, 1, 1, 0.3, 1], wspace=0.1) 

        # X ranges for the yellow shaded areas of the plots
        zoom_time_ranges = [(6, 8), (5, 7), (5, 7), (5,7), (0,1)]

    # Loop over the models to be plotted. Each column corresponds to a different model
    for (model_idx, model_key) in enumerate(models_for_plot):

        df_pred = models_parameters[model_key]['pred_trajectory']
        legend_colors.append(models_parameters[model_key]['color'])
        legend_labels.append(models_parameters[model_key]['name'])

        # For each model, loop over the state variables (x1, v1, x2, v2). Each state variable corresponds to a row in the plot grid
        for i, (var, ylabel) in enumerate(zip(variables, y_labels)):
            ax = fig.add_subplot(gs[i, model_idx])
            ax.plot(df_pred['time(s)'], df_pred[var], label=models_parameters[model_key]['name'], color=models_parameters[model_key]['color'], linewidth = 2)
            ax.plot(df_target['time(s)'], df_target[var], label='Target', linestyle="--", color="#2ca02c", linewidth = 2)
            
            # Highlight the zoomed-in region with yellow using the corresponding zoom range
            zoom_start, zoom_end = zoom_time_ranges[i]  
            ax.axvspan(zoom_start, zoom_end, color='yellow', alpha=0.3)
            
            # Only set ylabel for the first column
            if model_idx == 0:
                ax.set_ylabel(ylabel, fontsize=28)
            
            # Only set xlabel for the bottom row
            if i == 3:
                ax.set_xlabel('Time (s)', fontsize=28)
            
            # Remove x and y labels for other plots
            if model_idx != 0:
                ax.set_yticklabels([])
            if i != 3:
                ax.set_xticklabels([])

            ax.set_xticks(np.arange(0, np.max(df_target['time(s)']), 1))
            ax.tick_params(axis='both', labelsize=25)
            
            # Set y_axis limits
            y_min = min(df_pred[var].min(), df_target[var].min()) - 0.2
            y_max = max(df_pred[var].max(), df_target[var].max()) + 0.2
            ax_ticks = np.arange(y_min, y_max, (y_max - y_min) / 4)
            ax.set_yticks(ax_ticks)
            #ax.set_yticklabels([f'{tick:.1f}' for tick in ax_ticks])
            if model_idx == 0:
                ax.set_yticklabels([f'{tick:.1f}' for tick in ax_ticks])  # Only leftmost plots show y-tick labels

            """# Create secondary y-axis for Absolute Error
            ax2 = ax.twinx()
            ax2.plot(df_pred['time(s)'], models_parameters[model_key]['abs_err'], color='#7f7f7f', linewidth = 2.5)
            ax2.set_ylabel(fr'$\epsilon_{{\mathrm{{abs}}}}$ {y_labels_units[i]}', color='#7f7f7f', fontsize=40)
            ax2.tick_params(axis='y', labelcolor='#7f7f7f', labelsize=40)
            ax2_ticks = np.linspace(0, global_mape_max, num=3)
            ax2.set_yticks(ax2_ticks)
            # remove y-ticks of the second axis from all the plots except the one in the last column
            if model_idx != (num_models_for_plot - 1): # (-1) because the index starts in 0
                ax2.set_yticklabels([]) 
                ax2.set_ylabel(None)"""

            ##################### Plot in the last column a comparison between all the plots ######################

            # (+1) as an empty row for aesthetic reasons was added
            ax_compare = fig.add_subplot(gs[i, num_models_for_plot + 1])  

            # Use the zoom range specific to this row
            zoom_start, zoom_end = zoom_time_ranges[i]
            time_mask = (df_pred['time(s)'] >= zoom_start) & (df_pred['time(s)'] <= zoom_end)
            y_min = float('inf')
            y_max = float('-inf')
            for other_model_key in models_for_plot:
                other_pred = models_parameters[other_model_key]['pred_trajectory']
                ax_compare.plot(other_pred['time(s)'], other_pred[var], label=models_parameters[other_model_key]['name'], color=models_parameters[other_model_key]['color'], linestyle='-', linewidth=3)
                y_min = min(y_min, other_pred[var][time_mask].min(), df_target[var][time_mask].min())
                y_max = max(y_max, other_pred[var][time_mask].max(), df_target[var][time_mask].max())
            padding = 0.05 * (y_max - y_min)
            y_min -= padding
            y_max += padding
            ax_compare.plot(df_target['time(s)'], df_target[var], label='Target', linestyle="--", color="#2ca02c", linewidth=3)
            # Set the zoomed-in x-axis range for this row
            ax_compare.set_xlim([zoom_start, zoom_end])
            ax_compare.set_ylim([y_min, y_max])
            if i == 3:
                ax_compare.set_xlabel('Time (s)', fontsize=28)
            ax_compare.tick_params(axis='both', labelsize=26)


    # Create the list of legend handles, modifying the first element to have a dashed line
    legend_handles = [Line2D([0], [0], color=legend_colors[0], lw=10, linestyle='--')] + \
        [Line2D([0], [0], color=color, lw=10) for color in legend_colors[1:]]

    # Add the legend to the figure
    fig.legend(
        legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), 
        ncol=len(legend_labels), fontsize=30, frameon=False
    )

    # Save figure
    output_dir = config['plotting']['output_dir'] + "single_initial_condition/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_1b")
    savefig(save_path, pad_inches = 0.2)

