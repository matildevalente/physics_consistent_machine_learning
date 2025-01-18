import os
import csv
import pickle
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from src.ltp_system.plotter.Figure_6e import run_experiment_6e
from src.ltp_system.utils import savefig

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
        'name': 'NN projection',
        'color': '#d62728'  
    },
    'proj_pinn': {
        'name': 'PINN projection',
        'color': '#9A5092'  
    }
}

# create a configuration file for the chosen architecture
def generate_config_(config, hidden_sizes, options):
    return {
        'dataset_generation' : config['dataset_generation'],
        'data_prep' : config['data_prep'],
        'nn_model': {
            'APPLY_EARLY_STOPPING': options['APPLY_EARLY_STOPPING'],
            'RETRAIN_MODEL'      : options['RETRAIN_MODEL'],
            'hidden_sizes'       : hidden_sizes,
            'activation_fns'     : options['activation_fns'],
            'num_epochs'         : options['num_epochs'],
            'learning_rate'      : config['nn_model']['learning_rate'],
            'batch_size'         : config['nn_model']['batch_size'],
            'training_threshold' : config['nn_model']['training_threshold'],
            'n_bootstrap_models' : options['n_bootstrap_models'],
            'lambda_physics'     : config['nn_model']['lambda_physics'],       
            'patience'           : options['patience'],
            'alpha'              : options['alpha'],
            'checkpoints_dir'    : options['checkpoints_dir'],

        },
        'plotting': {
            'output_dir': config['plotting']['output_dir'],
            'PLOT_LOSS_CURVES': False,
            'PRINT_LOSS_VALUES': options['PRINT_LOSS_VALUES'],
            'palette': config['plotting']['palette'],
            'barplot_palette': config['plotting']['output_dir'],
        }
    }
    
# compute the number of weights and biases in the nn
def compute_parameters(layer_config):

    layer_config = [3] + layer_config + [17]

    hidden_layers = layer_config[1:-1]
    n_weights = sum(layer_config[i] * layer_config[i+1] for i in range(len(layer_config) - 1)) # x[layer_0] * x[layer_1] + ... + x[layer_N-1]*x[layer_N]
    n_biases = sum(hidden_layers)

    return n_weights + n_biases

# plot the histogram of the number of parameters for the nn architectures
def plot_histogram_with_params(architectures, options):
    min_parameters = compute_parameters([options['min_neurons_per_layer']] * options['n_hidden_layers'])
    max_parameters = compute_parameters([options['max_neurons_per_layer']] * options['n_hidden_layers'])
    
    # Calculate parameters for each architecture
    params_list = [compute_parameters(arch) for arch in architectures]
    plt.figure(figsize=(8, 6))
    plt.hist(params_list, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(min_parameters, color='red', linestyle='dashed', linewidth=2, label=  f'Min Parameters ({min_parameters})')
    plt.axvline(max_parameters, color='green', linestyle='dashed', linewidth=2, label=f'Max Parameters ({max_parameters})')
    plt.title(f'Histogram of Neural Network Architectures Parameters ({len(architectures)} architectures)')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Frequency')
    plt.legend()

    # Save figures
    output_dir = options['output_dir'] + "table_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"architectures_histogram.pdf")
    savefig(save_path, pad_inches=0.2)

# 
def generate_random_architectures(options):
    print("\nGenerating random NN architectures for Figure 6a.")
    min_neurons_per_layer = options['min_neurons_per_layer']
    max_neurons_per_layer = options['max_neurons_per_layer']
    n_architectures = options['n_steps']
    n_hidden_layers = options['n_hidden_layers']

    # Generate architectures based on scale preference
    if options.get('log_random_architectures', True):
        # Generate logarithmically spaced numbers
        architectures = np.logspace(
            np.log10(min_neurons_per_layer),
            np.log10(max_neurons_per_layer),
            n_architectures,
            dtype=int
        )
    else:
        # Generate linearly spaced numbers
        architectures = np.linspace(
            min_neurons_per_layer,
            max_neurons_per_layer,
            n_architectures,
            dtype=int
        )

    # Add specific architectures and sort
    #architectures = np.concatenate([architectures, [5, 10, 18]])
    architectures = np.unique(np.sort(architectures))  # Added unique to remove potential duplicates

    # Convert each element to a sublist of repeated neurons
    architectures_list = []
    for architecture in architectures:
        sublist = [architecture] * n_hidden_layers
        architectures_list.append(sublist)
        
    return architectures_list

#
def get_random_architectures(options, architectures_file_path):
    if options['RETRAIN_MODEL']:
        # generate random nn architectures
        random_architectures_list = generate_random_architectures(options)
        # save the nn architectures to a file
        with open(architectures_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(random_architectures_list)
    else:
        # load the nn architectures from a file
        random_architectures_list = []
        with open(architectures_file_path, mode='r') as file:
            reader = csv.reader(file)
            random_architectures_list = [row for row in reader]
        random_architectures_list = [[int(num) for num in sublist] for sublist in random_architectures_list]
    
    return random_architectures_list

# main function to run the experiment
def run_experiment_6a(config_original, large_dataset_path, options):
    ###################################### 1. SETUP AND DEFINITIONS ###################################
    output_features = config_original['dataset_generation']['output_features']
    index_output_features = [output_features.index(feature) for feature in options['extract_results_specific_outputs']]

    # define the file paths for the results
    table_dir = os.path.join(options['output_dir'], 'table_results')
    os.makedirs(table_dir, exist_ok=True)
    all_results_file_path = os.path.join(table_dir, 'all_outputs_mean_results.csv')
    specific_outputs_file_path = os.path.join(table_dir, 'specific_outputs_results.csv')
    ###################################################################################################

    ###################################### 2. GET RANDOM ARCHITECTURES  ###############################
    architectures_file_path = os.path.join(table_dir, 'architectures.csv')
    random_architectures_list = get_random_architectures(options, architectures_file_path)
    #print("random_architectures_list = ", random_architectures_list)
    plot_histogram_with_params(random_architectures_list, options)
    ###################################################################################################

    ###################################### 4. GET RESULTS FOR EACH ARCHITECTURE SIZE #######################
    # if the model is being retrained, compute the results for each dataset size
    if options['RETRAIN_MODEL']:
        # initialize the results list
        rows_mean_all_outputs = []
        rows_specific_outputs = []
        checkpoint_dir = options['checkpoints_dir']
        dataset_size = [options['dataset_size']]

        # iterate over the nn architectures
        for idx, hidden_sizes in enumerate(tqdm(random_architectures_list, desc="Evaluating Different Architectures")):

            # generate the config for the nn
            config_ = generate_config_(config_original, hidden_sizes, options)
            
            # Get the list of files to analyze
            options['hidden_sizes']    = hidden_sizes
            options['checkpoints_dir'] = checkpoint_dir + "architecture_" + str(idx) + "/"
            df_results_all_architecture_idx, df_results_specific_architecture_idx = run_experiment_6e(config_, large_dataset_path, dataset_size, options)

            # compute the number of parameters for the nn architecture
            num_params = compute_parameters(hidden_sizes)

            # append results for mean all outputs
            mapes_nn               = df_results_all_architecture_idx['nn_mapes'][0]
            uncertanties_mape_nn   = df_results_all_architecture_idx['nn_mape_uncertainties'][0]
            mapes_proj             = df_results_all_architecture_idx['proj_mapes'][0]
            uncertanties_mape_proj = df_results_all_architecture_idx['proj_mape_uncertainties'][0]
            rmses_nn               = df_results_all_architecture_idx['nn_rmses'][0]
            uncertanties_rmse_nn   = df_results_all_architecture_idx['nn_rmse_uncertainties'][0]
            rmses_proj             = df_results_all_architecture_idx['proj_rmses'][0]
            uncertanties_rmse_proj = df_results_all_architecture_idx['proj_rmse_uncertainties'][0]
            rows_mean_all_outputs.append([hidden_sizes, num_params, mapes_nn, uncertanties_mape_nn, mapes_proj, uncertanties_mape_proj, rmses_nn, uncertanties_rmse_nn, rmses_proj, uncertanties_rmse_proj])
            
            # append results for specific outputs
            if options['extract_results_specific_outputs'] is not None:
                # iterate over each output feature 
                for output_idx, output_feature in enumerate(index_output_features):
                    mapes_nn   = df_results_specific_architecture_idx['nn_mapes'][output_idx]
                    mapes_proj = df_results_specific_architecture_idx['proj_mapes'][output_idx]
                    rmses_nn   = df_results_specific_architecture_idx['nn_rmses'][output_idx]
                    rmses_proj = df_results_specific_architecture_idx['proj_rmses'][output_idx]

                    rows_specific_outputs.append([hidden_sizes, num_params, output_feature, mapes_nn, mapes_proj, rmses_nn, rmses_proj])

        # reset the options checkpoint_dir value
        options['checkpoints_dir'] = checkpoint_dir

        # create dataframe with mean of all outputs results
        cols_names = ['architectures', 'num_params', 'mapes_nn', 'uncertanties_mape_nn', 'mapes_proj', 'uncertanties_mape_proj', 'rmses_nn', ' uncertanties_rmse_nn', 'rmses_proj', 'uncertanties_rmse_proj']
        df_all_outputs = pd.DataFrame(rows_mean_all_outputs, columns = cols_names)
        df_all_outputs = df_all_outputs.sort_values(by='num_params', ascending=True)
        df_all_outputs.to_csv(all_results_file_path, index=False)

        # create dataframe with mean of all outputs results
        cols_names = ['architecture', 'num_params', 'output_feature', 'mapes_nn', 'mapes_proj', 'rmses_nn', 'rmses_proj']
        df_specific_outputs = pd.DataFrame(rows_specific_outputs, columns = cols_names)
        df_specific_outputs = df_specific_outputs.sort_values(by='num_params', ascending=True)
        df_specific_outputs.to_csv(specific_outputs_file_path, index=False)

        # return results
        return df_all_outputs, df_specific_outputs
    
    else:
        df_all_outputs = pd.read_csv(all_results_file_path)
        df_specific_outputs = pd.read_csv(specific_outputs_file_path)
        
        return df_all_outputs, df_specific_outputs

# Plot the results for the mean of all outputs
def Figure_6a_mean_all_outputs(options, df):

    # Plot MAPE for NN and NN projection
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel('Number of Parameters', fontsize=24)
    ax1.set_ylabel('MAPE (\%)', fontsize=24)
    ax1.set_xscale('log')
    ax1.plot(df['num_params'], df['mapes_nn'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['mapes_proj'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24, width = 2, length = 6)
    ax1.tick_params(axis='x', labelsize=24, width = 2, length = 6)
    #ax1.legend(loc='center right', fontsize=20)
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAPE Variation Rate (\%)', fontsize=24, color='#B8B42D')
    ax2.set_xscale('log')
    improvement_rate = (df['mapes_proj'] - df['mapes_nn']) / df['mapes_nn'] * 100
    ax2.plot(df['num_params'], improvement_rate, '-o', color='#B8B42D', label='Improvement Rate (%)') 
    ax2.axhline(0, color='#B8B42D', linestyle='--', linewidth=2)  # Horizontal line at y=0
    ax2.tick_params(axis='y', labelsize=24, colors='#B8B42D')  # Set tick color to #B8B42D
    ax2.spines['right'].set_color('#B8B42D')  
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    #fig.legend(loc='upper right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    # Save figure
    fig.tight_layout()
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_all_outputs_mape.pdf")
    fig.savefig(save_path, pad_inches=0.2, format='pdf', dpi=300, bbox_inches='tight')


    # Initialize the figure
    fig, ax1 = plt.subplots(figsize=(7, 5))
    # Plot RMSE for NN and NN projection
    ax1.set_xlabel('Number of Parameters', fontsize=24)
    ax1.set_ylabel('RMSE', fontsize=24, labelpad=10)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(2e-2, 30e-2)
    ax1.set_yticks([3e-2, 5e-2, 1e-1, 2e-1])
    ax1.set_yticklabels(['3', '5', '10', '20'])
        # Plot lines without error bars
    ax1.plot(df['num_params'], df['rmses_nn'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['rmses_proj'],'-o', color=models_parameters['proj_nn']['color'],linestyle='--', label='NN projection')
    # Add error bars to the plots
    #ax1.errorbar(df['num_params'], df['rmses_nn'], yerr=df['uncertanties_rmse_nn'],fmt='-o', color=models_parameters['NN']['color'], label='NN', capsize=5)
    #ax1.errorbar(df['num_params'], df['rmses_proj'], yerr=df['uncertanties_rmse_proj'],fmt='-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection', capsize=5)
    ax1.tick_params(axis='y', labelsize=24, width = 2, length = 6)
    ax1.tick_params(axis='x', labelsize=24, width = 2, length = 6)
    #ax1.legend(loc='right', fontsize=20)
    # Add scientific notation label at the top of the y-axis
    plt.minorticks_off()
    y_max = plt.gca().get_ylim()[1] * 3.62
    x_min = plt.gca().get_ylim()[0] - 0.05
    ax1.text(x_min, y_max, r'($\times10^{-2}$)', transform=plt.gca().transAxes, fontsize=24, ha='left', va='center')
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE Variation Rate (\%)', fontsize=24, color='#B8B42D')
    ax2.set_xscale('log')
    improvement_rate_rmse = (df['rmses_proj'] - df['rmses_nn']) / df['rmses_nn'] * 100
    ax2.plot(df['num_params'], improvement_rate_rmse, '-o', color='#B8B42D', label='Improvement Rate (%)')  # Set line color to gray
    ax2.tick_params(axis='y', labelsize=24, colors='#B8B42D', width = 2, length = 6)  # Set tick color to gray
    ax2.spines['right'].set_color('#B8B42D')  # Set color of the second y-axis spine to gray
    #fig.legend(loc='right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)
    # Save figure
    fig.tight_layout()
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_all_outputs_rmse.pdf")
    fig.savefig(save_path, pad_inches=0.2, format='pdf', dpi=300, bbox_inches='tight')

# Plot the results for the specific outputs
def Figure_6a_specific_outputs(options, df_all, df_specific, output_features_names):
    # Columns: architecture, num_params, output_feature, mapes_nn, mapes_proj, rmses_nn, rmses_proj

    # Create a grid of subplots for specific outputs
    n_outputs = len(df_specific['output_feature'].unique())
    n_cols = 3
    n_rows = (n_outputs + 1) // 2  # Ceiling division to handle odd number of plots
    # Initialize the figure with same style as MAPE plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5*n_rows))
    axes = axes.flatten()
    plot_idx = 0
    y_pos = [0.45, 0.35, 0.35]

    for output_idx in df_specific['output_feature'].unique():
        # Get data for specific output
        ax = axes[plot_idx]
        df_specific_output = df_specific[df_specific['output_feature'] == output_idx]
        
        # Set axis labels and scale
        ax.set_xlabel('Number of Parameters', fontsize=24, fontweight='bold')
        # Only add ylabel for first plot in each row
        if plot_idx % n_cols == 0:  
            ax.set_ylabel('MAPE (\%)', fontsize=24, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_yscale('log')
        
        # Make frame more visible
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Plot with thicker lines and larger markers
        line1, = ax.plot(df_specific_output['num_params'], df_specific_output['mapes_nn'], '-o', color=models_parameters['NN']['color'],linewidth=3, markersize=10)
        line2, = ax.plot(df_specific_output['num_params'], df_specific_output['mapes_proj'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--',linewidth=3, markersize=10)
        
        # Customize the plot
        ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=2, length=4)
        ax.set_title(f'{output_features_names[output_idx]}', fontsize=28, pad=15, fontweight='bold')
        ax.set_xscale('log')
        
        # Set y-axis range and ticks
        ax.set_ylim(0, 350)
        ax.set_yticks([1, 10, 100])  # Remove e0 since these are already the actual values
        ax.set_yticklabels(['1', '10', '100'])
        ax.minorticks_off()
        plot_idx += 1
    
    # Remove any extra subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])
    lines = [line1, line2]
    labels = ['NN', 'NN projection']
    # Add single legend at the top
    fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.36, 0.34), ncol=3, fontsize=24, frameon=True)
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "specific_outputs_mape.pdf")
    fig.savefig(save_path, pad_inches=0.3, format='pdf', dpi=300, bbox_inches='tight')


    # Initialize the figure with same style as RMSE plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5*n_rows))
    axes = axes.flatten()
    plot_idx = 0
    y_pos = [0.8, 0.8, 0.8]

    for output_idx in df_specific['output_feature'].unique():
        # Get data for specific output
        ax = axes[plot_idx]
        df_specific_output = df_specific[df_specific['output_feature'] == output_idx]
        
        # Set axis labels and scale
        ax.set_xlabel('Number of Parameters', fontsize=24)
        # Only add ylabel for first plot in each row
        if plot_idx % n_cols == 0:  
            ax.set_ylabel('RMSE', fontsize=24, fontweight='bold')
        else:
            ax.set_ylabel('')
        ax.set_yscale('log')
        
        # Make frame more visible
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Plot with thicker lines and larger markers
        line1, = ax.plot(df_specific_output['num_params'], df_specific_output['rmses_nn'], '-o', color=models_parameters['NN']['color'],linewidth=3, markersize=10)
        line2, = ax.plot(df_specific_output['num_params'], df_specific_output['rmses_proj'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--',linewidth=3, markersize=10)
        
        # Customize the plot
        ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=2, length=4)
        ax.set_title(f'{output_features_names[output_idx]}', fontsize=28, pad=15)
        ax.set_xscale('log')
        
        # Set y-axis range and ticks
        ax.set_ylim(1.5e-3, 600e-3)
        ax.set_yticks([2e-3, 10e-3, 100e-3, 400e-3])
        ax.set_yticklabels(['2', '10', '100', '400'])
        # Adjust y position based on plot index to ensure consistent placement
        ax.text(0, y_pos[plot_idx], r'($\times10^{-3}$)', transform=ax.get_yaxis_transform(), fontsize=20)
        
        plot_idx += 1
    
    # Remove any extra subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])
    # Add single legend at the top
    fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.36, 0.34), ncol=3, fontsize=24, frameon=True)
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "specific_outputs_rmse.pdf")
    fig.savefig(save_path, pad_inches=0.3, format='pdf', dpi=300, bbox_inches='tight')

