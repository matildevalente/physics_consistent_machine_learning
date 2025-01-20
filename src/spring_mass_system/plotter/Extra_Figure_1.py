import os
import copy
import torch
from tqdm import tqdm
import contextlib
import io
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from scipy import stats
import torch.nn as nn
from src.spring_mass_system.utils import set_seed, get_predicted_trajectory, get_target_trajectory, load_checkpoint, compute_parameters
from src.spring_mass_system.pinn import PhysicsInformedNN, train_pinn, plot_loss_curves_pinn #, optimize_pinn_architecture
from src.spring_mass_system.utils import figsize, savefig
import matplotlib.gridspec as gridspec


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


def get_trained_pinn(config: Dict[str, Any], preprocessed_data: Any) -> tuple:
    """
    Initialize and train a Physics-Informed Neural Network (PINN) model.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing model parameters
        preprocessed_data (Any): Preprocessed training data
        
    Returns:
        tuple: (trained PINN model, training losses dictionary)
    """
    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the PINN model and move it to the selected device
    pinn_model = PhysicsInformedNN(config).to(device)
    
    # Define the loss function (Mean Squared Error)
    loss_fn = nn.MSELoss()

    # Get learning rate from config and initialize Adam optimizer
    learning_rate = config['pinn_model']['learning_rate']
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=learning_rate)
    
    # Train the PINN model and collect training and validation losses
    losses = train_pinn(config, pinn_model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir = None)

    return pinn_model, losses


def _get_results_pinn_errors_vs_lambda(config, N_lambdas, preprocessed_data):
    """
    Analyze PINN performance across different physics loss weights (lambda values).
    
    This function either runs a new study of PINN performance across different lambda
    values or loads previously computed results.
    
    Args:
        config: Configuration dictionary containing model and training parameters
        N_lambdas (int): Number of lambda values to test
        preprocessed_data: Preprocessed training data
        
    Returns:
        pd.DataFrame: DataFrame containing loss metrics for each lambda value,
                     or None if loading results fails
    """

    if config['pinn_model']['RUN_LAMBDA_STUDY']:

        # array of uniformly generated N_lambdas between 0 and 1
        lambdas_arr = np.linspace(0, 1, N_lambdas)

        # perform a deep copy of config
        config_copy = copy.deepcopy(config)
        preprocessed_data_copy = copy.deepcopy(preprocessed_data)

        #
        config_copy['plotting']['PRINT_LOSS_VALUES'] = False
        config_copy['plotting']['PLOT_LOSS_CURVES'] = False

        # store results for each lambda value
        rows = []

        # loop over the different lambda_physics values and train different PINNs. Store the loss results
        for lambda_value in tqdm(lambdas_arr, desc="Evaluating different lambda_physics values", unit="lambda", leave=True, file=sys.stdout):
            
            # Use tqdm with a context manager to suppress internal function outputs
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):

                # update the value of lambda_value
                config_copy['pinn_model']['loss_physics_weight'] = lambda_value

                # train the physics informed pinn
                set_seed(42)
                _, pinn_losses = get_trained_pinn(config_copy, preprocessed_data_copy)

                # append the values of pinn losses 
                new_row = {
                    'lambda_physics_value': lambda_value, 
                    'loss_train_total'    : pinn_losses['train_losses'][-1],
                    'loss_train_physics'  : pinn_losses['train_physics_losses'][-1],
                    'loss_train_data'     : pinn_losses['train_data_losses'][-1],
                    'loss_val'            : pinn_losses['val_losses'][-1]
                }
                rows.append(new_row)

        # save the rows values in a dataframe
        df = pd.DataFrame(rows)

        # save the dataframe to a local directory
        output_dir = output_dir = config['plotting']['output_dir'] + "additional_results/"
        os.makedirs(output_dir, exist_ok=True)  
        csv_path = os.path.join(output_dir, "table_lambda_physics_errors.csv")
        df.to_csv(csv_path, index=False)

        return df


    else:
        try:
            # load the results from the local directory
            output_dir = config['plotting']['output_dir'] + "additional_results/"
            csv_path = os.path.join(output_dir, "table_lambda_physics_errors.csv")
            
            if not os.path.exists(csv_path):
                print(f"Results file not found at: {csv_path}")
                return None
                
            df = pd.read_csv(csv_path)
            return df

        except FileNotFoundError:
            print(f"Results file not found at: {csv_path}")
            return None
        except pd.errors.EmptyDataError:
            print(f"Empty CSV file found at: {csv_path}")
            return None
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return None


def plot_pinn_errors_vs_lambda(config, preprocessed_data, N_lambdas):

    # 
    df_pinn_errors = _get_results_pinn_errors_vs_lambda(config, N_lambdas, preprocessed_data)

    # plot the errors as a function of the lambda values
    # Use LaTeX for pgf
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(20, 10))

    # The 4th column is used as blank space
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1], wspace=0.3, hspace=0.5) 

    # X ranges for the yellow shaded areas of the plots
    zoom_lambda_ranges = [(0, 0.99), (0, 0.99), (0, 0.99), (0,0.99)]

    # Initialize with infinity values to find true min/max
    y_min_overall = float('inf')
    y_max_overall = float('-inf')
    
    # Iterate through each loss type to find overall min and max
    for loss_key in ['loss_train_total', 'loss_train_physics', 'loss_train_data', 'loss_val']:
        min_loss_i = df_pinn_errors[loss_key].min()
        max_loss_i = df_pinn_errors[loss_key].max()

        # Update overall minimum if current loss minimum is smaller
        if min_loss_i < y_min_overall:
            y_min_overall = min_loss_i
            
        # Update overall maximum if current loss maximum is larger
        if max_loss_i > y_max_overall:
            y_max_overall = max_loss_i
    
    # Loop over the models to be plotted. Each column corresponds to a different model
    for (loss_type_idx, loss_key) in enumerate(['loss_train_total', 'loss_train_physics', 'loss_train_data', 'loss_val']):
        # Set appropriate labels and titles based on the loss type
        if(loss_key == 'loss_train_total'):
            y_label = r'Train Loss Total ($\mathcal{L}_{\mathrm{train}}$)'
            title   = r'$\mathcal{L}_{\mathrm{train}}$'
        elif(loss_key == 'loss_train_physics'):
            y_label = r'Weighted $\mathcal{L}_{\mathrm{physics}}$'
            title   = r'$\lambda_{\mathrm{physics}} * \mathcal{L}_{\mathrm{physics}}$'
        elif(loss_key == 'loss_train_data'):
            y_label = r'Weighted $\mathcal{L}_{\mathrm{data}}$'
            title   = r'$(1 - \lambda_{\mathrm{physics}}) * \mathcal{L}_{\mathrm{data}}$'
        elif(loss_key == 'loss_val'):
            y_label = r'Validation Loss ($\mathcal{L}_{\mathrm{val}}$)'
            title   = r'$\mathcal{L}_{\mathrm{val}}$'
        
        ax = fig.add_subplot(gs[0, loss_type_idx])
        ax.plot(df_pinn_errors['lambda_physics_value'], df_pinn_errors[loss_key], label=y_label, linewidth = 5)
        
        # Highlight the zoomed-in region with yellow using the corresponding zoom range
        zoom_start, zoom_end = zoom_lambda_ranges[loss_type_idx]  
        ax.set_yscale('log')

        ax.axvspan(zoom_start, zoom_end, color='yellow', alpha=0.3)
        
        # Only set ylabel for the first column
        if loss_type_idx == 0: 
            ax.set_ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$', fontweight='bold', fontsize=30)
        
        # Only set xlabel 
        ax.set_xlabel(r'$\lambda_{\mathrm{physics}}$', fontsize=30)

        # Add title to the first plot of each column
        ax.set_title(title, fontsize=35, pad = 20)
        
        # Remove x and y labels for other plots
        if loss_type_idx != 0:
            ax.set_yticklabels([])

        # Set x_axis limits
        x_min = df_pinn_errors['lambda_physics_value'].min()
        x_max = df_pinn_errors['lambda_physics_value'].max()
        padding = 0.05 * (x_max - x_min)
        ax.set_xlim([x_min - padding, x_max + padding])
        
        # Set 4 evenly spaced ticks between 0 and 1
        ax.set_xticks([0, 0.3, 0.7, 1])
        
        # Set y_axis limits
        y_min_overall = max(1e-10, y_min_overall)  # Ensure minimum value is positive
        log_range = np.log10(y_max_overall) - np.log10(y_min_overall)
        padding_factor = 0.05  # 5% padding
        y_min_padded = y_min_overall * (10**(-padding_factor * log_range))
        y_max_padded = y_max_overall * (10**(padding_factor * log_range))
        ax.set_ylim([y_min_padded, y_max_padded])
        ax.tick_params(axis='both', labelsize=26)

        ##################### Plot in the last column a comparison between all the plots ######################
        # Create zoomed subplot
        ax_zoom = fig.add_subplot(gs[1, loss_type_idx])
        
        # Get zoom range for current row
        zoom_start, zoom_end = zoom_lambda_ranges[loss_type_idx]
        
        # Create mask for zoomed region
        lambda_mask = (df_pinn_errors['lambda_physics_value'] >= zoom_start) & \
                     (df_pinn_errors['lambda_physics_value'] <= zoom_end)
        
        # Get min and max values for current loss type in zoomed region
        masked_data = df_pinn_errors[lambda_mask][loss_key]
        current_min = masked_data.min()
        current_max = masked_data.max()
        
        # Update overall min/max while ensuring positive values for log scale
        y_min = 1e-8#max(1e-8, min(y_min_overall, current_min))
        y_max = current_max#max(y_max_overall, current_max)
        
        # Calculate padding in log space
        log_range = np.log10(y_max) - np.log10(y_min)
        padding_factor = 0.05
        y_min_padded = y_min * (10**(-padding_factor * log_range))
        y_max_padded = y_max * (10**(padding_factor * log_range))
        
        # Plot the data
        ax_zoom.plot(df_pinn_errors['lambda_physics_value'], 
                    df_pinn_errors[loss_key], 
                    linewidth=5)
        
        # Set log scale for y-axis
        ax_zoom.set_yscale('log')
        
        # Set axis limits
        padding = 0.05 * (zoom_end - zoom_start)
        ax_zoom.set_xlim([zoom_start - padding, zoom_end + padding])
        ax_zoom.set_ylim([y_min_padded, y_max_padded])
        
        # Set labels and ticks
        ax_zoom.set_xlabel(r'$\lambda_{\mathrm{physics}}$', fontsize=30)
        ax_zoom.tick_params(axis='both', labelsize=26)
        
        # Only set ylabel for the first column
        if loss_type_idx == 0:
            ax_zoom.set_ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$', 
                             fontweight='bold', 
                             fontsize=30)
            
        ax_zoom.set_title('Zoom-in', fontsize = 25)
        ax_zoom.set_xticks([0, 0.3, 0.7, 0.99])
        # Add grid for better readability in log scale
        #ax_zoom.grid(True, which="both", ls="-", alpha=0.2)
        #ax_zoom.grid(True, which="minor", ls="--", alpha=0.2)


    # save the plot to a local directory
    output_dir = config['plotting']['output_dir'] + "additional_results/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_pinn_errors_vs_lambda")
    savefig(save_path, pad_inches=0.2)
    