import os
import pandas as pd
import  numpy as np
import matplotlib as mpl
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src.spring_mass_system.utils import figsize, newfig, savefig

pgf_with_latex = {
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
    "font.size": 38,                    # Set global font size here
    "axes.labelsize": 38,               # Fontsize for x and y labels
    "xtick.labelsize": 45,              # Fontsize for x-tick labels
    "ytick.labelsize": 45,              # Fontsize for y-tick labels
    "legend.fontsize": 27.1,            # Fontsize for legend
    "figure.figsize": (9, 7)            # Default figure size
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

# Figures 1e: Plot of one trajectory given a dataframe
def plot_predicted_energies_vs_target(
        config: Dict[str, Any], 
        df_target: pd.DataFrame, 
        df_nn: pd.DataFrame, 
        df_pinn: pd.DataFrame, 
        df_proj_nn: pd.DataFrame, 
        df_proj_pinn: pd.DataFrame
    ):

    # Create plot using LaTeX for pgf
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig, ax  = plt.subplots(figsize=(9, 7))

    # Solid lines for base models (NN, PINN)
    ax.plot(df_target['time(s)'], df_target['E'], label="Target", linestyle="-", color="#2ca02c",  linewidth=10, markersize=50)
    ax.plot(df_nn['time(s)'], df_nn['E'], label="NN", linestyle="-", color=models_parameters['NN']['color'], linewidth=7)
    ax.plot(df_pinn['time(s)'], df_pinn['E'], label="PINN", linestyle="-", color=models_parameters['PINN']['color'], linewidth=7)
    ax.plot(df_proj_nn['time(s)'], df_proj_nn['E'], label="NN Projection", linestyle="--", color=models_parameters['proj_nn']['color'], linewidth=10)
    ax.plot(df_proj_pinn['time(s)'], df_proj_pinn['E'], label="PINN Projection", linestyle=":", color=models_parameters['proj_pinn']['color'], linewidth=10)

    # Set labels and font sizes
    ax.set_ylabel("Energy (J)", fontsize=38)
    ax.set_xlabel("Time (s)", fontsize=38)

    # Set specific x-ticks
    x_ticks = np.linspace(0, df_target['time(s)'].iloc[-1], 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{x_tick:.1f}" for x_tick in x_ticks], fontsize=45)

    # Adjust y-limits for energy plot
    y_max = max(df_pinn['E'].max(), df_nn['E'].max()) + 0.08
    y_min = min(df_pinn['E'].max(), df_target['E'].max()) - 0.1
    ax.set_ylim(y_min, 5.6)
    y_ticks = np.linspace(y_min, y_max, 4)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y_tick:.2f}" for y_tick in y_ticks], fontsize=45)  # Format with 2 decimal places

    # Adjust legend with thicker and shorter lines
    legend = ax.legend(
        loc="upper center", 
        fontsize=27.5, 
        ncol=2, 
        handlelength=2,  # Shorter line length
        handleheight=1,  # Control the height
        handletextpad=0.45  # Space between line and text
    )

    # Make lines in legend thicker
    for line in legend.get_lines():
        line.set_linewidth(12)  # Set the line thickness

    # Save figures
    output_dir = config['plotting']['output_dir'] + "single_initial_condition/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"energy_vs_target")
    savefig(save_path, pad_inches=0.2)
