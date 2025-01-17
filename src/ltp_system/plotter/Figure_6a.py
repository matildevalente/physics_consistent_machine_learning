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

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.ltp_system.utils import savefig
from src.ltp_system.utils import set_seed, load_dataset, load_config, split_dataset, select_random_rows
from src.ltp_system.data_prep import DataPreprocessor, LoadDataset, setup_dataset_with_preproprocessing_info
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
            'patience'           : options['patience'],
            'alpha'              : options['alpha'],
            'checkpoints_dir'    : options['checkpoints_dir']
        },
        'plotting': {
            'output_dir': options['results_dir'],
            'PLOT_LOSS_CURVES': False,
            'PRINT_LOSS_VALUES': options['PRINT_LOSS_VALUES'],
            'palette': config['plotting']['palette'],
            'barplot_palette': config['plotting']['output_dir'],
        }
    }
    
# train the nn for a chosen architecture or load the parameters if it has been trained 
def get_trained_nn(config, data_preprocessing_info, idx_arc, train_data, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join(config['nn_model']['checkpoints_dir'], f'architecture_{idx_arc}')  #os.path.join('output', 'ltp_system', 'checkpoints', 'different_architectures', f'architecture_{idx_arc}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config['nn_model']['RETRAIN_MODEL']:
        nn_models, nn_losses_dict, training_time = get_trained_bootstraped_models(config['nn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
        return nn_models, nn_losses_dict, device, training_time
    else:
        try:
            nn_models, _, hidden_sizes, activation_fns, training_time = load_checkpoints(config['nn_model'], NeuralNetwork, checkpoint_dir)
            return nn_models, hidden_sizes, activation_fns, training_time
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")

# the the mape and mape uncertainty of the nn aggregated model
def evaluate_model(index_output_features, normalized_model_predictions, normalized_targets):
    # perform a copy to avoid modifying the original arrays
    targets_norm_ = (np.array(normalized_targets)).copy()
    normalized_model_predictions_ = (np.array(normalized_model_predictions)).copy()

    # compute the MAPE and MAPE uncertainty with respect to target
    mape_all_outputs = np.mean(np.abs((targets_norm_ - normalized_model_predictions_) / targets_norm_)) * 100
    # compute the RMSE and RMSE uncertainty with respect to target
    rmse_all_outputs = np.sqrt(mean_squared_error(targets_norm_, normalized_model_predictions_))

    # compute the uncertainties
    mape_err_list = []
    rmse_err_list = []
    targets_norm_ = (np.array(normalized_targets)).copy()
    normalized_model_predictions_ = (np.array(normalized_model_predictions)).copy()
    for idx in range(len(normalized_model_predictions_[0])):
        # compute individual errors
        mape_err = np.abs((targets_norm_[:, idx] - normalized_model_predictions_[:, idx]) / targets_norm_[:, idx]) * 100
        rmse_err = np.sqrt(np.mean((targets_norm_[:, idx] - normalized_model_predictions_[:, idx]) ** 2))

        # append the errors to the lists
        rmse_err_list.append(rmse_err)
        mape_err_list.append(mape_err)

    # compute the uncertainties
    mape_uncertainty = np.std(mape_err_list) / np.sqrt(len(mape_err_list))
    rmse_uncertainty = np.std(rmse_err_list) / np.sqrt(len(rmse_err_list))


    # loop over the output features to compute the MAPE and RMSE for each output feature
    normalized_model_predictions_ = (np.array(normalized_model_predictions)).copy()
    targets_norm_ = (np.array(normalized_targets)).copy()
    if index_output_features is not None:
        rows_list = []
        for i in index_output_features:
            normalized_preds_i   = normalized_model_predictions_[:, i]
            normalized_targets_i = targets_norm_[:, i]

            # compute the MAPE of the i-th output feature with respect to target
            mape = np.mean(np.abs((normalized_targets_i - normalized_preds_i) / normalized_targets_i)) * 100

            # compute the RMSE of the i-th output feature with respect to target
            rmse = np.sqrt(mean_squared_error(normalized_targets_i, normalized_preds_i))

            rows_list.append([index_output_features, mape, rmse])


    return mape_all_outputs, mape_uncertainty, rmse_all_outputs, rmse_uncertainty, rows_list

# get the mape of the nn projected predictions
def evaluate_projection(index_output_features, normalized_model_predictions, normalized_targets, normalized_inputs, data_preprocessed, w_matrix):
    # perform a copy to avoid modifying the original arrays
    normalized_inputs_ = (np.array(normalized_inputs)).copy()
    normalized_targets_ = (np.array(normalized_targets)).copy()
    normalized_model_predictions_ = (np.array(normalized_model_predictions)).copy()

    # get the normalized projection predicitions of the model
    normalized_proj_predictions  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions_), torch.tensor(normalized_inputs_), data_preprocessed, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions = np.array(torch.tensor(np.stack(normalized_proj_predictions)))

    # perform a copy to avoid modifying the original arrays
    normalized_proj_predictions_ = (normalized_proj_predictions).copy()
    normalized_targets_ = (np.array(normalized_targets)).copy()

    # compute the mape and sem with respect to target
    mape_all_outputs = np.mean(np.abs((normalized_targets_ - normalized_proj_predictions_) / normalized_targets_)) * 100
    rmse_all_outputs = np.sqrt(np.mean((normalized_targets_ - normalized_proj_predictions_) ** 2))

    # compute the uncertainties
    mape_err_list = []
    rmse_err_list = []
    for idx in range(len(normalized_proj_predictions_[0])):
        # compute individual errors
        mape_err = np.abs((normalized_targets_[:, idx] - normalized_proj_predictions_[:, idx]) / normalized_targets_[:, idx]) * 100
        rmse_err = np.sqrt(np.mean((normalized_targets_[:, idx] - normalized_proj_predictions_[:, idx]) ** 2))

        # append the errors to the lists
        rmse_err_list.append(rmse_err)
        mape_err_list.append(mape_err)

    # compute the uncertainties
    mape_uncertainty = np.std(mape_err_list) / np.sqrt(len(mape_err_list))
    rmse_uncertainty = np.std(rmse_err_list) / np.sqrt(len(rmse_err_list))


    # loop over the output features to compute the MAPE and RMSE for each output feature
    if index_output_features is not None:
        rows_list = []
        for i in index_output_features:
            # copy to avoid modifying the original arrays
            normalized_proj_predictions_ = (np.array(normalized_model_predictions)).copy()
            normalized_targets_ = (np.array(normalized_targets)).copy()
            
            # get the i-th output feature
            normalized_preds_i   = normalized_proj_predictions_[:, i]
            normalized_targets_i = normalized_targets_[:, i]

            # compute the MAPE of the i-th output feature with respect to target
            mape = np.mean(np.abs((normalized_targets_i - normalized_preds_i) / normalized_targets_i)) * 100

            # compute the RMSE of the i-th output feature with respect to target
            rmse = np.sqrt(mean_squared_error(normalized_targets_i, normalized_preds_i))

            rows_list.append([index_output_features, mape, rmse])

    return mape_all_outputs, rmse_all_outputs, mape_uncertainty, rmse_uncertainty, rows_list

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
    output_dir = options['results_dir'] + "table_results"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"architectures_histogram.pdf")
    savefig(save_path, pad_inches=0.2)

# 
def get_random_architectures(options):
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
    
    architectures_list.append([451, 315, 498, 262])

    return architectures_list

# split the dataset into training and testing
def split_dataset_(config_, large_dataset_path, n_testing_points):
    """
    METHODOLOGY: 
    (1) Begin with a large dataset (N points). 
    (2) Select the size of the testing dataset size (n_testing_points = 300). 
    (3) Create testing  dataset: Randomly select "n_testing_points" points from the large_dataset. All the datasets should be tested on this dataset.
    (4) Create training dataset: Randomly select N_train points from the large dataset. This large dataset should not include the testing points to avoid data leakage.
    (5) Preprocess the data.
    """
    # create the preprocessed_data object - all the smaller datasets should use the same scalers
    _, large_dataset = load_dataset(config_, large_dataset_path)
    data_preprocessing_info = DataPreprocessor(config_)
    data_preprocessing_info.setup_dataset(large_dataset.x, large_dataset.y)  

    # separate the test set of the large_dataset_path from the rest of the dataset which will be used to train the various models
    testing_file, training_file = split_dataset(large_dataset_path, n_testing_points)
    test_inputs_norm, test_targets_norm = load_data(testing_file, data_preprocessing_info)

    return data_preprocessing_info, training_file, test_inputs_norm, test_targets_norm



# main function to run the experiment
# (config, large_dataset_path, options_fig_6a, dataset_size = 1000, seed = 42)
def run_experiment_6a(config_original, large_dataset_path, options, n_testing_points = 500):
    set_seed(42) 
    ###################################### 1. SETUP AND DEFINITIONS ###################################
    # define the file paths for the results
    table_dir = os.path.join(options['results_dir'], 'table_results')
    # create the directory if it doesn't exist
    os.makedirs(table_dir, exist_ok=True)
    architectures_file_path = os.path.join(table_dir, 'architectures.csv')
    all_results_file_path = os.path.join(table_dir, 'all_outputs_mean_results.csv')
    specific_outputs_file_path = os.path.join(table_dir, 'specific_outputs_results.csv')
    output_features = config_original['dataset_generation']['output_features']
    index_output_features = [output_features.index(feature) for feature in options['extract_results_specific_outputs']]
    ###################################################################################################

    ###################################### 2. GET RANDOM ARCHITECTURES  ###############################
    if options['RETRAIN_MODEL']:
        # generate random nn architectures
        random_architectures_list = get_random_architectures(options)
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

    print("random_architectures_list = ", random_architectures_list)
    plot_histogram_with_params(random_architectures_list, options)
    ###################################################################################################

    ####### 3. SELECT THE TESTING POINTS FROM LARGE DATASET AND STORE DATA_PREPROCESSING INFO #########
    data_preprocessing_info, training_file, test_inputs_norm, test_targets_norm = split_dataset_(config_original, large_dataset_path, n_testing_points)
    # Save the data_preprocessing_info object
    os.makedirs(options['checkpoints_dir'], exist_ok=True)
    file_path = os.path.join(options['checkpoints_dir'], "data_preprocessing_info.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(data_preprocessing_info, file)
    ###################################################################################################

    ###################################### 4. GET RANDOM ARCHITECTURES  ###############################
    # 1. read from the train_path file and randomly select 'dataset_size' rows - save the dataset to a local dir sampled_dataset_dir
    sampled_dataset_dir = select_random_rows(training_file, options['dataset_size'], seed = 42)
    # 2. read the dataset from the sampled_dataset_dir and preprocess data
    _, sampled_dataset = load_dataset(config_original, sampled_dataset_dir)
    # 3. preprocess the data: the training sets, ie, subsets of the bigger dataset, are preprocessed using the scalers fitted on the large dataset to avoid data leakage.
    train_data_norm, _, val_data_norm  = setup_dataset_with_preproprocessing_info(sampled_dataset.x, sampled_dataset.y, data_preprocessing_info)  
    # 4. create the val loader needed for training the NN.
    val_loader = torch.utils.data.DataLoader(val_data_norm, batch_size=config_original['nn_model']['batch_size'], shuffle=True)
    ###################################################################################################

    ###################################### 4. GET RESULTS FOR EACH ARCHITECTURE SIZE #######################
    if options['RETRAIN_MODEL']:
        # initialize the results list
        results = []
        outputs_specific_results = []
        # iterate over the nn architectures
        for idx, hidden_sizes in enumerate(tqdm(random_architectures_list, desc="Evaluating Different Architectures")):
            set_seed(42) 
            # 1. generate the activation functions for the nn
            activation_fns = [options['activation_func']] * len(hidden_sizes) 

            # 2. generate the config for the nn
            config_ = generate_config_(config_original, hidden_sizes, activation_fns, options)

            # 3. train model and count training time
            nn_models, _, _, _ = get_trained_nn(config_, data_preprocessing_info, idx, train_data_norm, val_loader)
            
            # 4. perform copies of the test inputs and test targets to avoid modifying them.
            test_inputs_norm_  = test_inputs_norm.clone() 
            test_targets_norm_ = test_targets_norm.clone() 

            # 5. use the trained nn to make predictions on the test inputs - get the normalized model predictions 
            nn_predictions_norm =  get_average_predictions(nn_models, torch.tensor(test_inputs_norm_))
            nn_pred_uncertainties =  get_predictive_uncertainty(nn_models, torch.tensor(test_inputs_norm_)) # for each point prediction gives an uncertainty value
            
            # 6. perform copies of the test inputs and test targets to avoid modifying them.
            nn_predictions_norm_  = nn_predictions_norm.clone() 
            nn_pred_uncertainties_ = nn_pred_uncertainties.clone() 
            test_targets_norm_ = test_targets_norm.clone() 
            
            # 7. compute errors of the nn predictions on the test set: mape_all_outputs, mape_uncertainty, rmse_all_outputs, rmse_uncertainty, rows_list
            mape_nn, mape_uncertainty_nn, rmse_nn, rmse_uncertainty_nn, results_output_list_nn  = evaluate_model(index_output_features, nn_predictions_norm_, test_targets_norm_)

            # 8. compute errors of the proj predictions on the test set:perform model predictions with the trained nn
            mape_proj, rmse_proj, mape_proj_uncertainty, rmse_proj_uncertainty, results_output_list_proj = evaluate_projection(index_output_features, nn_predictions_norm_, test_targets_norm_, test_inputs_norm_, data_preprocessing_info, options['w_matrix'])

            # compute the number of parameters for the nn architecture
            params = compute_parameters(hidden_sizes)

            # append the results for the whole dataset
            results.append((params, mape_nn, mape_uncertainty_nn, mape_proj, mape_proj_uncertainty, rmse_nn, rmse_uncertainty_nn, rmse_proj, rmse_proj_uncertainty))

            # append the results for the specific outputs.   # index of output,           mape nn        ,            rmse nn        ,             mape proj     ,            rmse proj
            outputs_specific_results.append((params, results_output_list_nn, results_output_list_proj))
        
        # /// 4. CREATE THE DATAFRAME FOR THE RESULTS CONCERNING THE WHOLE DATASET ///
        params, mapes_nn, mape_uncertainties_nn, mapes_proj, mape_proj_uncertainties, rmses_nn, rmse_uncertainties_nn, rmses_proj, rmse_proj_uncertainties = zip(*results)
        data = {
            'architectures': random_architectures_list, 
            'num_params': params,
            'mapes_nn': mapes_nn,
            'uncertanties_mape_nn': mape_uncertainties_nn,
            'mapes_proj': mapes_proj,
            'uncertanties_mape_proj': mape_proj_uncertainties,
            'rmses_nn': rmses_nn, 
            'uncertanties_rmse_nn': rmse_uncertainties_nn,
            'rmses_proj': rmses_proj,
            'uncertanties_rmse_proj': rmse_proj_uncertainties,
        }
        # Create a DataFrame from the results and store as .csv in a local directory
        df_all = pd.DataFrame(data)
        df_all = df_all.sort_values(by='num_params', ascending=True)
        df_all.to_csv(all_results_file_path, index=False)
        

        # /// 5. CREATE THE DATAFRAME FOR THE RESULTS CONCERNING THE SPECIFIC OUTPUTS ///
        if options['extract_results_specific_outputs'] is not None:
            # initialize the results list
            params, results_output_list_nn, results_output_list_proj  = zip(*outputs_specific_results)
            specific_outputs_rows = []

            # iterate over the nn architectures and number of parameters
            for architecture_idx, (architecture, param) in enumerate(zip(random_architectures_list, params)):
                
                # iterate over each output feature 
                for output_idx, output_feature in enumerate(index_output_features):

                    # for each architecture and output feature, get the mape and rmse of the nn and the nn projection
                    mape_nn = results_output_list_nn[architecture_idx][output_idx][1]
                    rmse_nn = results_output_list_nn[architecture_idx][output_idx][2]
                    mape_proj = results_output_list_proj[architecture_idx][output_idx][1]
                    rmse_proj = results_output_list_proj[architecture_idx][output_idx][2]

                    row = [architecture, param, output_feature, mape_nn, mape_proj, rmse_nn, rmse_proj]

                    # append the rows to the data list
                    specific_outputs_rows.append(row)

            # Create a DataFrame from the results and store as .csv in a local directory
            cols_names = ['architecture', 'num_params', 'output_feature', 'mapes_nn', 'mapes_proj', 'rmses_nn', 'rmses_proj']
            df_specific_outputs = pd.DataFrame(specific_outputs_rows, columns = cols_names)
            df_specific_outputs = df_specific_outputs.sort_values(by='num_params', ascending=True)
            df_specific_outputs.to_csv(specific_outputs_file_path, index=False)

        return df_all, df_specific_outputs
    else:
        df_all = pd.read_csv(all_results_file_path)
        df_specific_outputs = pd.read_csv(specific_outputs_file_path)

        return df_all, df_specific_outputs

# Plot the results for the mean of all outputs
def Figure_6a_mean_all_outputs(options, df):

    # Plot MAPE for NN and NN projection
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel('Number of Parameters', fontsize=24)
    ax1.set_ylabel('MAPE (\%)', fontsize=24)
    ax1.plot(df['num_params'], df['mapes_nn'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['mapes_proj'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.legend(loc='upper right', fontsize=20)
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
    output_dir = options['results_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_all_outputs_mape.pdf")
    fig.savefig(save_path, pad_inches=0.2, format='pdf', dpi=300, bbox_inches='tight')


    # Initialize the figure
    fig, ax1 = plt.subplots(figsize=(7, 5))
    # Plot RMSE for NN and NN projection
    ax1.set_xlabel('Number of Parameters', fontsize=24)
    ax1.set_ylabel('RMSE', fontsize=24, labelpad=10)
    ax1.set_yscale('log')
    ax1.set_ylim(5e-2, 30e-2)
    ax1.set_yticks([5e-2, 1e-1, 2e-1])
    ax1.set_yticklabels(['5', '10', '20'])
        # Plot lines without error bars
    ax1.plot(df['num_params'], df['rmses_nn'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['rmses_proj'],'-o', color=models_parameters['proj_nn']['color'],linestyle='--', label='NN projection')
    # Add error bars to the plots
    #ax1.errorbar(df['num_params'], df['rmses_nn'], yerr=df['uncertanties_rmse_nn'],fmt='-o', color=models_parameters['NN']['color'], label='NN', capsize=5)
    #ax1.errorbar(df['num_params'], df['rmses_proj'], yerr=df['uncertanties_rmse_proj'],fmt='-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection', capsize=5)
    ax1.tick_params(axis='y', labelsize=24)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.legend(loc='right', fontsize=20)
    # Add scientific notation label at the top of the y-axis
    plt.minorticks_off()
    y_max = plt.gca().get_ylim()[1] * 3.62
    x_min = plt.gca().get_ylim()[0] - 0.05
    ax1.text(x_min, y_max, r'($\times10^{-2}$)', transform=plt.gca().transAxes, fontsize=24, ha='left', va='center')
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE Variation Rate (\%)', fontsize=24, color='gray')
    improvement_rate_rmse = (df['rmses_proj'] - df['rmses_nn']) / df['rmses_nn'] * 100
    ax2.plot(df['num_params'], improvement_rate_rmse, '-o', color='gray', label='Improvement Rate (%)')  # Set line color to gray
    ax2.tick_params(axis='y', labelsize=24, colors='gray')  # Set tick color to gray
    ax2.spines['right'].set_color('gray')  # Set color of the second y-axis spine to gray
    #fig.legend(loc='right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    # Save figure
    fig.tight_layout()
    output_dir = options['results_dir'] + "plots"
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
        
        # Set y-axis range and ticks
        ax.set_ylim(8, 150)
        ax.set_yticks([10, 20, 40, 80])  # Remove e0 since these are already the actual values
        ax.set_yticklabels(['10', '20', '40', '80'])
        ax.minorticks_off()
        plot_idx += 1
    
    # Remove any extra subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])
    lines = [line1, line2]
    labels = ['NN', 'NN projection']
    # Add single legend at the top
    fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, 0.34), ncol=3, fontsize=24, frameon=True)
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend
    output_dir = options['results_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "specific_outputs_mape.pdf")
    fig.savefig(save_path, pad_inches=0.3, format='pdf', dpi=300, bbox_inches='tight')


    # Initialize the figure with same style as RMSE plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5*n_rows))
    axes = axes.flatten()
    plot_idx = 0
    y_pos = [0.45, 0.45, 0.45]

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
        
        # Set y-axis range and ticks
        ax.set_ylim(1.5e-2, 40e-2)
        ax.set_yticks([2e-2, 5e-2, 10e-2, 20e-2])
        ax.set_yticklabels(['2', '5', '10', '20'])
        # Adjust y position based on plot index to ensure consistent placement
        ax.text(0, y_pos[plot_idx], r'($\times10^{-2}$)', transform=ax.get_yaxis_transform(), fontsize=20)
        
        plot_idx += 1
    
    # Remove any extra subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])
    # Add single legend at the top
    fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, 0.34), ncol=3, fontsize=24, frameon=True)
    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend
    output_dir = options['results_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "specific_outputs_rmse.pdf")
    fig.savefig(save_path, pad_inches=0.3, format='pdf', dpi=300, bbox_inches='tight')

