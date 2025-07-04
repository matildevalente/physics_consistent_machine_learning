import os
import io
import sys
import ast
import copy
import torch
import contextlib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from src.spring_mass_system.utils import savefig
from src.spring_mass_system.utils import set_seed
from src.ltp_system.pinn_nn import get_trained_bootstraped_models



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


def get_trained_pinn(config, seed, data_preprocessing_info, train_data, val_loader):
    # create checkpoints directory for the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'pinn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # either train the model from scratch ...
    pinn_models, pinn_losses_dict, _ = get_trained_bootstraped_models(config['pinn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed)
    return pinn_models, pinn_losses_dict, device
    

def _run_results_pinn_errors_vs_lambda(train_data, val_loader, config, lambdas_arr, preprocessed_data, n_seeds, case):
    """
    Analyze PINN performance across different physics loss weights (lambda values).
    
    This function either runs a new study of PINN performance across different lambda
    values or loads previously computed results.
    """

    print("Lambda values =\n", np.array2string(lambdas_arr, precision=17, suppress_small=False))

    # perform a deep copy of config
    config_copy = copy.deepcopy(config)
    preprocessed_data_copy = copy.deepcopy(preprocessed_data)

    #
    config_copy['plotting']['PLOT_LOSS_CURVES'] = False
    config_copy['pinn_model']['n_bootstrap_models'] = 1

    # store results for each lambda value
    rows = []

    # loop over the different lambda_physics values and train different PINNs. Store the loss results
    for lambda_value in tqdm(lambdas_arr, desc="Evaluating different lambda_physics values", unit="lambda", leave=True, file=sys.stdout):
        
        # Use tqdm with a context manager to suppress internal function outputs
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):

            train_data_copy = copy.deepcopy(train_data)
            val_loader_copy  = copy.deepcopy(val_loader)

            # update the value of lambda_value
            config_copy['pinn_model']['lambda_physics'] = lambda_value.tolist()
            
            
            loss_train_total_list, loss_train_physics_list, loss_train_data_list, loss_val_list, loss_val_physics_list = [], [], [], [], []
            loss_train_P_list, loss_train_I_list, loss_train_ne_list, loss_val_P_list, loss_val_I_list, loss_val_ne_list = [], [], [], [], [], []
            for seed in range(n_seeds):
                # train the physics informed pinn
                set_seed(seed+1)
                trained_pinn_models, pinn_losses_dict, device = get_trained_pinn(config_copy, seed+1, preprocessed_data_copy, train_data_copy, val_loader_copy)

                # get mean errors on training data
                loss_train_total_list.append(pinn_losses_dict['losses_train_mean'][-1])
                loss_train_data_list.append(pinn_losses_dict['losses_train_data_mean'][-1])
                loss_train_physics_list.append(pinn_losses_dict['losses_train_physics_mean'][-1])
                loss_train_P_list.append(pinn_losses_dict['losses_train_P_mean'][-1])
                loss_train_I_list.append(pinn_losses_dict['losses_train_I_mean'][-1])
                loss_train_ne_list.append(pinn_losses_dict['losses_train_ne_mean'][-1])
                

                # get mean errors on validation data
                loss_val_list.append(pinn_losses_dict['losses_val_mean'][-1])
                loss_val_physics_list.append(pinn_losses_dict['losses_val_physics_mean'][-1])
                loss_val_P_list.append(pinn_losses_dict['losses_val_P_mean'][-1])
                loss_val_I_list.append(pinn_losses_dict['losses_val_I_mean'][-1])
                loss_val_ne_list.append(pinn_losses_dict['losses_val_ne_mean'][-1])

            # append the values of pinn losses 
            new_row = {
                'lambda_physics_value'  : lambda_value, 
                'loss_train_total'      : np.mean(loss_train_total_list),
                'loss_train_total_err'  : np.std(loss_train_total_list) / np.sqrt(n_seeds),
                'loss_train_data'       : np.mean(loss_train_data_list),
                'loss_train_data_err'   : np.std(loss_train_data_list) / np.sqrt(n_seeds),
                'loss_train_physics'    : np.mean(loss_train_physics_list),
                'loss_train_physics_err': np.std(loss_train_physics_list) / np.sqrt(n_seeds),
                'loss_train_P'    : np.mean(loss_train_P_list),
                'loss_train_P_err': np.std(loss_train_P_list) / np.sqrt(n_seeds),
                'loss_train_I'    : np.mean(loss_train_I_list),
                'loss_train_I_err': np.std(loss_train_I_list) / np.sqrt(n_seeds),
                'loss_train_ne'    : np.mean(loss_train_ne_list),
                'loss_train_ne_err': np.std(loss_train_ne_list) / np.sqrt(n_seeds),
                'loss_val'              : np.mean(loss_val_list), 
                'loss_val_err'          : np.std(loss_val_list) / np.sqrt(n_seeds),
                'loss_val_physics'      : np.mean(loss_val_physics_list), 
                'loss_val_physics_err'  : np.std(loss_val_physics_list) / np.sqrt(n_seeds), 
                'loss_val_P'      : np.mean(loss_val_P_list), 
                'loss_val_P_err'  : np.std(loss_val_P_list) / np.sqrt(n_seeds), 
                'loss_val_I'      : np.mean(loss_val_I_list), 
                'loss_val_I_err'  : np.std(loss_val_I_list) / np.sqrt(n_seeds), 
                'loss_val_ne'      : np.mean(loss_val_ne_list), 
                'loss_val_ne_err'  : np.std(loss_val_ne_list) / np.sqrt(n_seeds)
            }
            rows.append(new_row)

    # save the rows values in a dataframe
    df = pd.DataFrame(rows)

    # save the dataframe to a local directory
    output_dir = output_dir = config['plotting']['output_dir'] + "additional_results/"
    os.makedirs(output_dir, exist_ok=True)  
    csv_path = os.path.join(output_dir, f'table_lambda_physics_errors_{case}.csv')
    df.to_csv(csv_path, index=False, float_format='%.17e')


def _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case, data_set):
    
    # Create figure and primary axis
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig, ax1 = plt.subplots(figsize=(8, 6))
    #ax2 = ax1.twinx() # Create second y-axis
    
    # get the summed values of lambda physics across the physical laws
    lambda_values = df_pinn_errors['lambda_physics_value'].apply(lambda s: np.sum(ast.literal_eval(s.replace(" ", ",")))).to_numpy()

    # Get lambda values
    if case == "near_1":
        lambda_values = 1 - lambda_values
        x_label = r'$(1 - \lambda_{\mathrm{physics}})$'
    else:
        x_label = r'$\lambda_{\mathrm{physics}}$'
    
    if data_set == "train":
        y_label_1 = r'$\mathcal{L}^{\mathrm{train}}$'

        # get loss values
        loss_data = df_pinn_errors['loss_train_data']
        loss_data_err = df_pinn_errors['loss_train_data_err']
        
        loss_physics_total = df_pinn_errors['loss_train_physics']
        loss_physics_total_err = df_pinn_errors['loss_train_physics_err']
        
        loss_P_total = df_pinn_errors['loss_train_P']
        loss_P_total_err = df_pinn_errors['loss_train_P_err']
        
        loss_I_total = df_pinn_errors['loss_train_I']
        loss_I_total_err = df_pinn_errors['loss_train_I_err']
            
        loss_ne_total = df_pinn_errors['loss_train_ne']
        loss_ne_total_err = df_pinn_errors['loss_train_ne_err']

        loss_train_total = df_pinn_errors['loss_train_total']
        loss_train_total_err = df_pinn_errors['loss_train_total_err']
    

    elif data_set == "val":
        y_label_1 = r'$\mathcal{L}$'#^{\mathrm{val}}$'

        loss_data = df_pinn_errors['loss_val']
        loss_data_err = df_pinn_errors['loss_val_err']
        
        loss_physics_total = df_pinn_errors['loss_val_physics']
        loss_physics_total_err = df_pinn_errors['loss_val_physics_err']
        
        loss_P_total = df_pinn_errors['loss_val_P']
        loss_P_total_err = df_pinn_errors['loss_val_P_err']
        
        loss_I_total = df_pinn_errors['loss_val_I']
        loss_I_total_err = df_pinn_errors['loss_val_I_err']
            
        loss_ne_total = df_pinn_errors['loss_val_ne']
        loss_ne_total_err = df_pinn_errors['loss_val_ne_err']

    # Equation (9)
    ax1.plot(lambda_values, loss_data, linewidth=4, color='#2ca02c', label=r'$\mathcal{L}_{\mathrm{data}}^{\mathrm{val}}$')
    ax1.errorbar(lambda_values, loss_data, yerr=loss_data_err,color='#2ca02c',alpha=0.6, capsize=3,  capthick=2, elinewidth=2,fmt='none' )
    
    # Equation (15)
    ax1.plot(lambda_values, loss_physics_total, linewidth=4, color='#d62728', label=r'$\mathcal{L}_{\mathrm{physics}}^{\mathrm{val}}$')
    ax1.errorbar(lambda_values, loss_physics_total,  yerr=loss_physics_total_err,color='#d62728',alpha=0.6, capsize=3,capthick=2, elinewidth=2, fmt='none')
    
    # Equation (11)
    ax1.plot(lambda_values, loss_P_total, linewidth=4, color='#1f77b4', label=r'$\mathcal{L}_{\mathrm{P}}$')
    ax1.errorbar(lambda_values, loss_P_total,  yerr=loss_P_total_err,color='#1f77b4',alpha=0.6, capsize=3,capthick=2, elinewidth=2, fmt='none')
    
    # Equation (12)
    ax1.plot(lambda_values, loss_I_total, linewidth=4, color='#9300ad', label=r'$\mathcal{L}_{\mathrm{I}}$')
    ax1.errorbar(lambda_values, loss_I_total,  yerr=loss_I_total_err,color="#9300ad",alpha=0.6, capsize=3,capthick=2, elinewidth=2, fmt='none')
    
    # Equation (13)
    ax1.plot(lambda_values, loss_ne_total, linewidth=4, color='#d6a800', label=r'$\mathcal{L}_{\mathrm{n_e}}$')
    ax1.errorbar(lambda_values, loss_ne_total,  yerr=loss_ne_total_err,color="#d6a800",alpha=0.6, capsize=3,capthick=2, elinewidth=2, fmt='none')
    
    # Equation (10)
    if data_set == "train":
        ax1.plot(lambda_values, loss_train_total, linewidth=4, color="#414040", label=r'$\mathcal{L}_{\mathrm{total}}^{\mathrm{train}}$', linestyle='--',)
        ax1.errorbar(lambda_values, loss_train_total, yerr=loss_train_total_err,color='#414040',alpha=1, capsize=3,  capthick=2, elinewidth=2,fmt='none' )


    # Gather handles and labels from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    custom_handles = [
        Line2D([0], [0], color=h.get_color(), linewidth=5)  # thicker legend line
        for h in handles1
    ]

    # Add a combined legend outside the plot in two columns
    fig.legend(custom_handles, labels1, loc='upper center', bbox_to_anchor=(0.6, 1.27), ncol=3, fontsize=35, frameon=False, handlelength=1,handletextpad=0.5)


    if case == "near_1":

        plt.xscale('log')
        ax1.set_yscale('log')

        ax1.set_xticks([1e-8, 1e-6, 1e-3])
        ax1.set_xticklabels([r'$10^{-8}$', r'$10^{-6}$', r'$10^{-3}$'])

        ax1.set_yticks([1e-3,1e-1, 1e1])
        ax1.set_yticklabels([r'$10^{-3}$',r'$10^{-1}$',  r'$10^{1}$'])

    elif(case == "uniform"):

        ax1.set_yscale('log')

        ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_xticklabels([r'$0$', r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1$'])
        ax1.set_xlim([-0.1, 1.1])

        ax1.set_yticks([ 1e-3, 1e-1, 1e1])
        ax1.set_yticklabels([r'$10^{-3}$',r'$10^{-1}$',  r'$10^{1}$'])
    
    elif(case == "near_0"):
        
        plt.xscale('log')
        ax1.set_yscale('log')

        ax1.set_xticks([1e-5, 1e-3, 1e-1])
        ax1.set_xticklabels([ r'$10^{-5}$', r'$10^{-3}$', r'$10^{-1}$'])

        ax1.set_yticks([1e-5, 1e-3, 1e-1])
        ax1.set_yticklabels([ r'$10^{-5}$', r'$10^{-3}$', r'$10^{-1}$'])

    # Set tick parameters for styling
    ax1.set_ylabel(y_label_1, fontweight='bold', fontsize=35, color="#000000", labelpad = 12)
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


def _get_results_pinn_errors_vs_lambda(config, case):
    try:
        # load the results from the local directory
        output_dir = config['plotting']['output_dir'] + "additional_results/"
        csv_path = os.path.join(output_dir, f'table_lambda_physics_errors_{case}.csv')
        
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


def plot_pinn_errors_vs_lambda(config, preprocessed_data, train_data, val_loader, print_messages=True):
    if print_messages:
        print("\n\n====================  Lambda_Physics vs. Validation Loss Study  ====================")


    if config['pinn_model']['RUN_LAMBDA_STUDY']:
        # each model is initilized with a different seed - the final error is the mean of the errors 
        n_seeds = 20

        # array of uniformly generated N_lambdas between 0 and 1 and run the results 
        x = np.linspace(0, 1, num=30) / 3
        lambdas_arr = np.stack([x, x, x], axis=1)
        _run_results_pinn_errors_vs_lambda(train_data, val_loader, config, lambdas_arr, preprocessed_data, n_seeds, case = "uniform")

        # 
        x = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]) 
        lambdas_arr = np.stack([x, x, x], axis=1)
        _run_results_pinn_errors_vs_lambda(train_data, val_loader, config, lambdas_arr, preprocessed_data, n_seeds, case = "near_0")

        # 
        x = np.array([1- 1e-3,1- 1e-4,1- 1e-5,1- 1e-6,1- 1e-7,1- 1e-8, 1- 1e-9, 1- 1e-10, 1- 1e-11, 1- 1e-12, 1- 1e-13, 1- 1e-14, 1- 1e-15], dtype=np.float64) / 3
        lambdas_arr = np.stack([x, x, x], axis=1)
        _run_results_pinn_errors_vs_lambda(train_data, val_loader, config, lambdas_arr, preprocessed_data, n_seeds, case = "near_1")
    

    try: 
        # get the results from the directory and plot of validation loss as a function of lambda
        df_pinn_errors = _get_results_pinn_errors_vs_lambda(config, case = "uniform")
        _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "uniform", data_set = "val")
        _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "uniform", data_set = "train")
        
        # get the results from the directory and plot of validation loss as a function of lambda
        df_pinn_errors = _get_results_pinn_errors_vs_lambda(config, case = "near_0")
        _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "near_0", data_set = "val")
        _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "near_0", data_set = "train")
        
        # get the results from the directory and plot of validation loss as a function of lambda
        df_pinn_errors = _get_results_pinn_errors_vs_lambda(config, case = "near_1")
        _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "near_1", data_set = "val")
        _plot_pinn_errors_vs_lambda_simple(config, df_pinn_errors, case = "near_1", data_set = "train")
    
    except Exception as e:
        print(f"Error loading results: {str(e)}")
        print("Please turn the 'RUN_LAMBDA_STUDY' flag in the pinn_model configurations to True if you want to see the results of the validation loss as a function of lambda_physics.")
        print(f"Continuing without this analysis with the pre-defined lambda_physics = { config['pinn_model']['lambda_physics'] }.")
        return None

