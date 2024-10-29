import os
import csv
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.ltp_system.utils import savefig
from src.ltp_system.utils import set_seed, load_dataset, load_config
from src.ltp_system.data_prep import DataPreprocessor, LoadDataset
from src.ltp_system.pinn_nn import get_trained_bootstraped_models, load_checkpoints, NeuralNetwork, get_average_predictions, get_predictive_uncertainty
from src.ltp_system.projection import get_average_predictions_projected,constraint_p_i_ne

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

# load and preprocess the test dataset from a local directory
def load_data(test_filename, preprocessed_data):
    # load and extract experimental dataset
    test_dataset = LoadDataset(test_filename)
    test_targets, test_inputs = test_dataset.y, test_dataset.x

    # apply log transform to the skewed features
    if len(preprocessed_data.skewed_features_in) > 0:
        test_inputs[:, preprocessed_data.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, preprocessed_data.skewed_features_in]))

    if len(preprocessed_data.skewed_features_out) > 0:
        test_targets[:, preprocessed_data.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, preprocessed_data.skewed_features_out]))

    # 3. normalize targets with the model used on the training data
    normalized_inputs  = torch.cat([torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) for i, scaler in enumerate(preprocessed_data.scalers_input)], dim=1)
    normalized_targets = torch.cat([torch.from_numpy(scaler.transform(test_targets[:, i:i+1])) for i, scaler in enumerate(preprocessed_data.scalers_output)], dim=1)

    return normalized_inputs, normalized_targets

# create a configuration file for the chosen architecture
def generate_config_(config, hidden_sizes, activation_fns, options):
    return {
        'dataset_generation' : config['dataset_generation'],
        'data_prep' : config['data_prep'],
        'nn_model': {
            'APPLY_EARLY_STOPPING': options['APPLY_EARLY_STOPPING'],
            'RETRAIN_MODEL'      : options['RETRAIN_MODEL'],
            'hidden_sizes'       : hidden_sizes,
            'activation_fns'     : activation_fns,
            'num_epochs'         : options['num_epochs'],
            'learning_rate'      : config['nn_model']['learning_rate'],
            'batch_size'         : config['nn_model']['batch_size'],
            'training_threshold' : config['nn_model']['training_threshold'],
            'n_bootstrap_models' : options['n_bootstrap_models'],
            'lambda_physics'     : config['nn_model']['lambda_physics'],            
        },
        'plotting': {
            'output_dir': config['plotting']['output_dir'],
            'PLOT_LOSS_CURVES': False,
            'PRINT_LOSS_VALUES': options['PRINT_LOSS_VALUES'],
            'palette': config['plotting']['palette'],
            'barplot_palette': config['plotting']['output_dir'],
        }
    }
    
# train the nn for a chosen architecture or load the parameters if it has been trained 
def get_trained_nn(config, preprocessed_data, idx_arc):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'different_architectures', f'architecture_{idx_arc}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_data = preprocessed_data.train_data
    val_loader = torch.utils.data.DataLoader(preprocessed_data.val_data, batch_size=config['nn_model']['batch_size'], shuffle=True)

    if config['nn_model']['RETRAIN_MODEL']:
        nn_models, nn_losses_dict, training_time = get_trained_bootstraped_models(config['nn_model'], config['plotting'], preprocessed_data, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
        return nn_models, nn_losses_dict, device, training_time
    else:
        try:
            nn_models, _, hidden_sizes, activation_fns, training_time = load_checkpoints(config['nn_model'], NeuralNetwork, checkpoint_dir)
            return nn_models, hidden_sizes, activation_fns, training_time
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")


# the the mape and mape uncertainty of the nn aggregated model
def evaluate_model(nn_models, normalized_inputs, normalized_targets):
    # perform a copy to avoid modifying the original arrays
    normalized_inputs_ = (np.array(normalized_inputs)).copy()
    targets_norm_ = (np.array(normalized_targets)).copy()
    
    # get the normalized model predictions 
    normalized_model_predictions =  get_average_predictions(nn_models, torch.tensor(normalized_inputs_))
    model_pred_uncertainty =  get_predictive_uncertainty(nn_models, torch.tensor(normalized_inputs_)) 

    m = len(model_pred_uncertainty)

    #
    normalized_model_predictions_ = (np.array(normalized_model_predictions)).copy()
    model_pred_uncertainty_ = (np.array(model_pred_uncertainty)).copy()

    # compute the mape and sem with respect to target
    mape = np.mean(np.abs((targets_norm_ - normalized_model_predictions_) / targets_norm_)) * 100
    mape_uncertainty = np.sqrt(np.mean(np.square(model_pred_uncertainty_ / targets_norm_))) * 100

    rmse = np.sqrt(mean_squared_error(normalized_targets, normalized_model_predictions_))
    rmse_uncertainty = (1 / np.sqrt(m)) * np.sqrt(np.sum(np.square(model_pred_uncertainty_)))

    return mape, mape_uncertainty, normalized_model_predictions_, rmse, rmse_uncertainty


# get the mape of the nn projected predictions
def evaluate_projection(normalized_model_predictions, normalized_targets, normalized_inputs, data_preprocessed, w_matrix):
    # perform a copy to avoid modifying the original arrays
    normalized_inputs_ = (np.array(normalized_inputs)).copy()
    normalized_targets_ = (np.array(normalized_targets)).copy()
    normalized_model_predictions_ = (np.array(normalized_model_predictions)).copy()
    
    # get the normalized projection predicitions of the model
    normalized_proj_predictions  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions_), torch.tensor(normalized_inputs_), data_preprocessed, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions = np.array(torch.tensor(np.stack(normalized_proj_predictions)))

    # compute the mape and sem with respect to target
    mape = np.mean(np.abs((normalized_targets_ - normalized_proj_predictions) / normalized_targets_)) * 100
    rmse = np.sqrt(np.mean((normalized_targets_ - normalized_proj_predictions) ** 2))

    return mape, rmse


# compute the number of weights and biases in the nn
def compute_parameters(layer_config):

    layer_config = [3] + layer_config + [17]

    hidden_layers = layer_config[1:-1]
    n_weights = sum(layer_config[i] * layer_config[i+1] for i in range(len(layer_config) - 1)) # x[layer_0] * x[layer_1] + ... + x[layer_N-1]*x[layer_N]
    n_biases = sum(hidden_layers)

    return n_weights + n_biases

#
def plot_histogram_with_params(architectures, options, config_plotting):
    min_parameters = compute_parameters([options['min_neurons_per_layer']] * options['min_hidden_layers'])
    max_parameters = compute_parameters([options['max_neurons_per_layer']] * options['max_hidden_layers'])
    
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
    output_dir = config_plotting['output_dir']  + "Figures_6a"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_6a_extra")
    savefig(save_path, pad_inches = 0.2)

#
def get_random_architectures(options):

    print("\nGenerating random NN architectures for Figure 6a.")
    min_neurons_per_layer = options['min_neurons_per_layer']
    max_neurons_per_layer = options['max_neurons_per_layer']
    n_architectures = options['n_steps']
    max_hidden_layers = options['max_hidden_layers']
    min_hidden_layers = options['min_hidden_layers']


    # Generate uniformly distributed number of hidden layers (as integers)
    architectures = np.linspace(min_neurons_per_layer, max_neurons_per_layer, n_architectures, dtype=int)
    architectures = np.concatenate([architectures, [5, 10, 18]])
    architectures = np.sort(architectures)

    # Convert each element of hidden_layers_list to a sublist of repeated neurons
    architectures_list = []
    for architecture in architectures:
        sublist = [architecture] * min_hidden_layers
        architectures_list.append(sublist)

    return architectures_list


# main function to run the experiment
def run_experiment_6a(config_original, filename, options):
    # /// 1. EXTRACT DATASET & PREPROCESS THE DATASET///
    _, full_dataset = load_dataset(config_original, dataset_dir = filename)
    preprocessed_data = DataPreprocessor(config_original)
    preprocessed_data.setup_dataset(full_dataset.x, full_dataset.y)  
    normalized_inputs, normalized_targets = load_data( filename, preprocessed_data)
    
    architectures_file_path = 'output/ltp_system/checkpoints/different_architectures/architectures.csv'
    results_file_path = 'output/ltp_system/checkpoints/different_architectures/results.csv'

    if options['RETRAIN_MODEL']:
        random_architectures_list = get_random_architectures(options)
        with open(architectures_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(random_architectures_list)
    else:
        random_architectures_list = []
        with open(architectures_file_path, mode='r') as file:
            reader = csv.reader(file)
            random_architectures_list = [row for row in reader]
        random_architectures_list = [[int(num) for num in sublist] for sublist in random_architectures_list]

    print(random_architectures_list)
    plot_histogram_with_params(random_architectures_list, options, config_original['plotting'])
    
    if options['RETRAIN_MODEL']:
        results = []
        for idx, hidden_sizes in enumerate(tqdm(random_architectures_list, desc="Evaluating Different Architectures")):

            activation_fns = [options['activation_func']] * len(hidden_sizes) 

            config_ = generate_config_(config_original, hidden_sizes, activation_fns, options)

            # train model and count training time
            nn_models, _, _, training_time = get_trained_nn(config_, preprocessed_data, idx)

            # perform model predictions with the trained nn
            mape_nn, mape_uncertainty_nn, normalized_model_predictions, rmse_nn, rmse_uncertainty_nn  = evaluate_model(nn_models, normalized_inputs, normalized_targets)

            # perform model predictions with the trained nn
            mape_proj, rmse_proj = evaluate_projection(normalized_model_predictions, normalized_targets, normalized_inputs, preprocessed_data, options['w_matrix'])

            params = compute_parameters(hidden_sizes)
            results.append((params, mape_nn, mape_uncertainty_nn, mape_proj, rmse_nn, rmse_uncertainty_nn, rmse_proj, training_time))

        params, mapes_nn, mape_uncertainties_nn, mapes_proj, rmses_nn, rmse_uncertainties_nn, rmses_proj, times = zip(*results)
        data = {
            'architectures': random_architectures_list, 
            'num_params': params,
            'mapes_nn': mapes_nn,
            'uncertanties_mape_nn': mape_uncertainties_nn,
            'mapes_proj': mapes_proj,
            'rmses_nn': rmses_nn, 
            'uncertanties_rmse_nn': rmse_uncertainties_nn,
            'rmses_proj': rmses_proj,
            'model_training_time': times
        }
        # Create a DataFrame from the results and store as .csv in a local directory
        df = pd.DataFrame(data)
        df = df.sort_values(by='num_params', ascending=True)
        df.to_csv(results_file_path, index=False)
        
        return df
    else:
        df = pd.read_csv(results_file_path)
        return df

# Plot the results
def Figure_6a(config_plotting, df):

    # Plot MAPE for NN and NN projection
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Number of Parameters', fontsize=24)
    ax1.set_ylabel('MAPE (\%)', fontsize=24)
    ax1.plot(df['num_params'], df['mapes_nn'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['mapes_proj'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.legend(loc='upper right', fontsize=24)
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAPE Variation Rate (\%)', fontsize=24, color='gray')
    improvement_rate = (df['mapes_proj'] - df['mapes_nn']) / df['mapes_nn'] * 100
    ax2.plot(df['num_params'], improvement_rate, '-o', color='gray', label='Improvement Rate (%)') 
    ax2.axhline(0, color='lightgray', linestyle='--', linewidth=2)  # Horizontal line at y=0
    ax2.tick_params(axis='y', labelsize=24, colors='gray')  # Set tick color to gray
    ax2.spines['right'].set_color('gray')  # Set color of the second y-axis spine to gray
    #fig.legend(loc='upper right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    # Save figure
    fig.tight_layout()
    output_dir = config_plotting['output_dir'] + "Figures_6a"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Figure_6a_mape")
    fig.savefig(save_path, pad_inches=0.2)


    # Initialize the figure
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Plot RMSE for NN and NN projection
    ax1.set_xlabel('Number of Parameters', fontsize=24)
    ax1.set_ylabel('RMSE', fontsize=24)
    ax1.set_yscale('log')
    ax1.set_ylim(5e-2, 6e-1)
    ax1.set_yticks([5e-2, 1e-1, 2e-1, 4e-1])
    ax1.set_yticklabels(['5', '10', '20', '40'])
    ax1.plot(df['num_params'], df['rmses_nn'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['rmses_proj'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.legend(loc='right', fontsize=24)
    # Add scientific notation label at the top of the y-axis
    ax1.text(plt.gca().get_position().bounds[0] - 0.13, 1.05, r'$\times 10^{-2}$', 
            transform=plt.gca().transAxes, fontsize=24, ha='left', va='center')
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE Variation Rate (\%)', fontsize=24, color='gray')
    improvement_rate_rmse = (df['rmses_proj'] - df['rmses_nn']) / df['rmses_nn'] * 100
    ax2.plot(df['num_params'], improvement_rate_rmse, '-o', color='gray', label='Improvement Rate (%)')  # Set line color to gray
    ax2.tick_params(axis='y', labelsize=24, colors='gray')  # Set tick color to gray
    ax2.spines['right'].set_color('gray')  # Set color of the second y-axis spine to gray
    fig.legend(loc='right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    # Save figure
    fig.tight_layout()
    output_dir = config_plotting['output_dir'] + "Figures_6a"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "Figure_6a_rmse")
    fig.savefig(save_path, pad_inches=0.2)
    plt.show()


