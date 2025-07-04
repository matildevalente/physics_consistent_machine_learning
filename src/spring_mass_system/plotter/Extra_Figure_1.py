import os
import io
import sys
import copy
import torch
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.spring_mass_system.utils import savefig
from src.spring_mass_system.utils import set_seed
from src.spring_mass_system.pinn_nn import NeuralNetwork, train_model



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


def get_trained_pinn(config, preprocessed_data, seed):
    """
    Initialize and train a Physics-Informed Neural Network (PINN) model.
    """
    # Set the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the PINN model and move it to the selected device
    pinn_model = NeuralNetwork(config['pinn_model']).to(device)

    # Get learning rate from config and initialize Adam optimizer
    learning_rate = config['pinn_model']['learning_rate']
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=learning_rate)
    
    # Train the PINN model and collect training and validation losses
    losses = train_model(config, config['pinn_model'], pinn_model, preprocessed_data, optimizer, device, None, seed)

    return pinn_model, losses, device


def _get_results_pinn_errors_vs_lambda(config, lambdas_arr, preprocessed_data, n_seeds, scale):
    """
    Analyze PINN performance across different physics loss weights (lambda values).
    
    This function either runs a new study of PINN performance across different lambda
    values or loads previously computed results.
    """

    if config['pinn_model']['RUN_LAMBDA_STUDY']:

        print("Lambda values = ", lambdas_arr)

        # perform a deep copy of config
        config_copy = copy.deepcopy(config)
        preprocessed_data_copy = copy.deepcopy(preprocessed_data)

        #
        config_copy['plotting']['PRINT_LOSS_VALUES'] = True
        config_copy['plotting']['PLOT_LOSS_CURVES'] = False

        # store results for each lambda value
        rows = []

        # loop over the different lambda_physics values and train different PINNs. Store the loss results
        for lambda_value in tqdm(lambdas_arr, desc="Evaluating different lambda_physics values", unit="lambda", leave=True, file=sys.stdout):
            
            # Use tqdm with a context manager to suppress internal function outputs
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):

                # update the value of lambda_value
                config_copy['pinn_model']['loss_physics_weight'] = lambda_value
                
                loss_train_total_list, loss_train_physics_list, loss_train_data_list, loss_val_list, loss_val_physics_list = [], [], [], [], []
                for seed in range(n_seeds):
                    # train the physics informed pinn
                    set_seed(seed+1)
                    _, pinn_losses, _ = get_trained_pinn(config_copy, preprocessed_data_copy, seed+1)

                    # get mean errors
                    loss_train_total_list.append(pinn_losses['train_losses'][-1])
                    loss_train_physics_list.append(pinn_losses['train_physics_losses'][-1])
                    loss_train_data_list.append(pinn_losses['train_data_losses'][-1])
                    loss_val_list.append(pinn_losses['val_data_losses'][-1])
                    loss_val_physics_list.append(pinn_losses['val_physics_losses'][-1])

                # append the values of pinn losses 
                new_row = {
                    'lambda_physics_value'  : lambda_value, 
                    'loss_train_total'      : np.mean(loss_train_total_list),
                    'loss_train_total_err'  : np.std(loss_train_total_list) / np.sqrt(n_seeds),
                    'loss_train_physics'    : np.mean(loss_train_physics_list),
                    'loss_train_physics_err': np.std(loss_train_physics_list) / np.sqrt(n_seeds),
                    'loss_train_data'       : np.mean(loss_train_data_list),
                    'loss_train_data_err'   : np.std(loss_train_data_list) / np.sqrt(n_seeds),
                    'loss_val'              : np.mean(loss_val_list), 
                    'loss_val_err'          : np.std(loss_val_list) / np.sqrt(n_seeds),
                    'loss_val_physics'      : np.mean(loss_val_physics_list), 
                    'loss_val_physics_err'  : np.std(loss_val_physics_list) / np.sqrt(n_seeds)
                }
                rows.append(new_row)

        # save the rows values in a dataframe
        df = pd.DataFrame(rows)

        # save the dataframe to a local directory
        output_dir = output_dir = config['plotting']['output_dir'] + "additional_results/"
        os.makedirs(output_dir, exist_ok=True)  
        csv_path = os.path.join(output_dir, f'table_lambda_physics_errors_{scale}.csv')
        df.to_csv(csv_path, index=False)

        return df


    else:
        try:
            # load the results from the local directory
            output_dir = config['plotting']['output_dir'] + "additional_results/"
            csv_path = os.path.join(output_dir, f'table_lambda_physics_errors_{scale}.csv')
            
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


def _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case, data_set):
    
    # Create figure and primary axis
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    #ax2 = ax1.twinx() # Create second y-axis
    
    # Get lambda values
    if case == "near_1":
        lambda_values = 1 - df_pinn_errors['lambda_physics_value']
        x_label = r'$(1 - \lambda_{\mathrm{physics}})$'
    else:
        lambda_values = df_pinn_errors['lambda_physics_value']
        x_label = r'$\lambda_{\mathrm{physics}}$'
    
    if data_set == "train":
        y_label_1 = r'$\mathcal{L}$'
        legend_1 = r'$\mathcal{L}_{\mathrm{data}}^{train}$'
        legend_2 =  r'$\mathcal{L}_{\mathrm{physics}}^{train}$'

        # get loss values
        loss_data = df_pinn_errors['loss_train_data']
        loss_data_err = df_pinn_errors['loss_train_data_err']
        
        loss_energy = df_pinn_errors['loss_train_physics']
        loss_energy_err = df_pinn_errors['loss_train_physics_err']

        loss_train_total = df_pinn_errors['loss_train_total']
        loss_train_total_err = df_pinn_errors['loss_train_total_err']

    elif data_set == "val":
        y_label_1 = r'$\mathcal{L}$' #r'$\mathcal{L}_{\mathrm{Val}}$'
        legend_1 = r'$\mathcal{L}_{\mathrm{data}}^{val}$'
        legend_2 =  r'$\mathcal{L}_{\mathrm{physics}}^{val}$'

        loss_data = df_pinn_errors['loss_val']
        loss_data_err = df_pinn_errors['loss_val_err']
        
        loss_energy = df_pinn_errors['loss_val_physics']
        loss_energy_err = df_pinn_errors['loss_val_physics_err']

    # Equation (9)
    ax1.plot(lambda_values, loss_data, linewidth=4, color='#2ca02c', label=legend_1)
    ax1.errorbar(lambda_values, loss_data, yerr=loss_data_err,color='#2ca02c',alpha=0.6, capsize=3,  capthick=2, elinewidth=2,fmt='none' )
    
    # Equation (15)
    ax1.plot(lambda_values, loss_energy, linewidth=4, color='#d62728', label=legend_2)
    ax1.errorbar(lambda_values, loss_energy,  yerr=loss_energy_err,color='#d62728',alpha=0.6, capsize=3,capthick=2, elinewidth=2, fmt='none')
    
    # Equation (10)
    if data_set == "train":
        ax1.plot(lambda_values, loss_train_total, linewidth=4, color="#414040", label=r'$\mathcal{L}_{\mathrm{total}}^{train}$', linestyle='--',)
        #ax1.errorbar(lambda_values, loss_train_total, yerr=loss_train_total_err,color='#414040',alpha=1, capsize=3,  capthick=2, elinewidth=2,fmt='none' )


    # Gather handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    custom_handles = [
        Line2D([0], [0], color=h.get_color(), linewidth=5)  # thicker legend line
        for h in handles1
    ]

    # Add a combined legend outside the plot in two columns
    fig.legend(custom_handles, labels1, loc='upper center', bbox_to_anchor=(0.6, 1.25), ncol=3, fontsize=35, frameon=False, handlelength=1)


    if case == "near_1":

        plt.xscale('log')
        ax1.set_yscale('log')

        if data_set == "val":
            ax1.set_yticks([ 1e-6, 1e-4, 1e-2, 1e1])
            ax1.set_yticklabels([r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$',  r'$10^{1}$'])
            ax1.set_ylim([5e-8, 4e-2])
    
        else:
            ax1.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e1])
            ax1.set_yticklabels([r'$10^{-10}$', r'$10^{-8}$', r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$',  r'$10^{1}$'])
            ax1.set_ylim([1e-11, 4e-2])

        #ax1.set_xticks([1e-10,1e-7, 1e-4, 1e-2])
        #ax1.set_xticklabels([r'$10^{-10}$',r'$10^{-7}$', r'$10^{-4}$', r'$10^{-2}$'])
        ax1.set_xticks([1e-6, 1e-4, 1e-2])
        ax1.set_xticklabels([r'$10^{-6}$',r'$10^{-4}$', r'$10^{-2}$'])
        ax1.set_xlim([5e-7, 0.3])


    elif(case == "uniform"):

        ax1.set_yscale('log')

        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_xticklabels([r'$0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1$'])
        ax1.set_xlim([-0.1, 1.1])
        
        ax1.set_yticks([ 1e-8, 1e-5, 1e-2, 1e1])
        ax1.set_yticklabels([r'$10^{-8}$',r'$10^{-5}$',r'$10^{-2}$',  r'$10^{1}$'])
    

    # Set tick parameters for styling
    ax1.set_ylabel(y_label_1, fontweight='bold', fontsize=45, color="#000000", labelpad = 12)
    ax1.set_xlabel(x_label, fontsize=40)
    ax1.tick_params(axis='x', labelsize=40, length=8, width=1.5)
    ax1.tick_params(axis='y', labelcolor="#000000", labelsize=40, length=8, width=1.5)
    
    # Save plot if output directory provided
    plt.tight_layout()
    output_dir = config['plotting']['output_dir'] + "additional_results/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_loss_vs_lambda_{data_set}_{case}")
    savefig(save_path, pad_inches=0.2)
    print(f"\nPlot of the Loss values as a function of the Lambda parameter of the PINN model saved as .pdf file to:\n   → {output_dir}.")


def plot_pinn_errors_vs_lambda(config, preprocessed_data, N_lambdas, print_messages=True):
    if print_messages:
        print("\n\n====================  Lambda_Physics vs. Validation Loss Study  ====================")

    # 
    n_seeds = 20

    # array of uniformly generated N_lambdas between 0 and 1
    lambdas_arr = np.linspace(0, 1, num=30)
    df_pinn_errors = _get_results_pinn_errors_vs_lambda(config, lambdas_arr, preprocessed_data, n_seeds, scale = "lin")
    _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "uniform", data_set = "val")
    _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "uniform", data_set = "train")

    # array of uniformly generated N_lambdas between 0 and 1
    lambdas_arr = np.array([1-10**-1, 1-10**-2, 1-10**-3,1-10**-4,1-10**-5,1-10**-6])
    df_pinn_errors = _get_results_pinn_errors_vs_lambda(config, lambdas_arr, preprocessed_data, n_seeds, scale = "log")
    _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "near_1", data_set = "val")
    _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "near_1", data_set = "train")

