import os
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
        'name': 'NN Projection',
        'color': '#d62728'  
    },
    'proj_pinn': {
        'name': 'PINN Projection',
        'color': '#9A5092'  
    }
}

# load and preprocess the test dataset from a local directory
def load_data(test_filename, data_preprocessing_info):
    # load and extract experimental dataset
    test_dataset = LoadDataset(test_filename)
    test_targets, test_inputs = test_dataset.y, test_dataset.x

    # apply log transform to the skewed features
    if len(data_preprocessing_info.skewed_features_in) > 0:
        test_inputs[:, data_preprocessing_info.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, data_preprocessing_info.skewed_features_in]))

    if len(data_preprocessing_info.skewed_features_out) > 0:
        test_targets[:, data_preprocessing_info.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, data_preprocessing_info.skewed_features_out]))

    # 3. normalize targets with the model used on the training data
    normalized_inputs  = torch.cat([torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) for i, scaler in enumerate(data_preprocessing_info.scalers_input)], dim=1)
    normalized_targets = torch.cat([torch.from_numpy(scaler.transform(test_targets[:, i:i+1])) for i, scaler in enumerate(data_preprocessing_info.scalers_output)], dim=1)
 
    return normalized_inputs, normalized_targets

# create a configuration file for the chosen architecture
def generate_config_(config, hidden_sizes, activation_fns, options):
    return {
        'dataset_generation' : config['dataset_generation'],
        'data_prep' : {
            'fraction_train': 0.8,    
            'fraction_val': 0.1,    
            'fraction_test': 0.1,    
            'skew_threshold_down': 0 , 
            'skew_threshold_up': 3,
        },
        'nn_model': {
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
def get_trained_nn(config, data_preprocessing_info, idx_dataset, train_data, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'different_datasets', f'dataset_{idx_dataset}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    if config['nn_model']['RETRAIN_MODEL']:
        nn_models, nn_losses_dict, training_time = get_trained_bootstraped_models(config['nn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data)

        return nn_models, nn_losses_dict, device, training_time
    else:
        try:
            nn_models, _, hidden_sizes, activation_fns, training_time = load_checkpoints(config['nn_model'], NeuralNetwork, checkpoint_dir)
            return nn_models, hidden_sizes, activation_fns, training_time
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")

# compute the approximate loki time required to generate the dataset
def compute_approximate_loki_time(dataset_size, loki_reference_time):

    """
    This function computes an approximate runtime for LoKI simulations based on the size of the dataset.

    Parameters:
    - dataset_size: The number of simulations in the current dataset.
    - loki_reference_time: The average time (in seconds) required to run 100 LoKI simulations.

    The approximation assumes that the runtime scales linearly with the number of simulations. 
    Therefore, the total runtime for a dataset of any size is computed by scaling the reference time for 100 simulations.
    """

    return ( dataset_size * loki_reference_time ) / 100


# the the mape and mape uncertainty of the nn aggregated model
def evaluate_model(model_predictions_norm, model_pred_uncertainties, targets_norm):

    targets_norm_ = targets_norm.clone() 
    model_pred_uncertainties_ = model_pred_uncertainties.clone()
    m = len(model_pred_uncertainties)

    # compute the mape and the uncertainty 
    mape_j = np.mean(np.abs((targets_norm_ - model_predictions_norm) / targets_norm_)) * 100
    sigma_mape_j = np.sqrt(np.mean(np.square(model_pred_uncertainties_ / targets_norm_))) * 100
    
    # compute the rmse and the uncertainty 
    rmse_j = np.sqrt(np.mean((targets_norm_ - model_predictions_norm) ** 2))
    sigma_rmse_j = (1 / np.sqrt(m)) * np.sqrt(np.sum(np.square(model_pred_uncertainties_)))

    return mape_j, sigma_mape_j, rmse_j, sigma_rmse_j

# get the mape of the nn projected predictions
def evaluate_projection(normalized_model_predictions, normalized_targets, normalized_inputs, data_preprocessed, w_matrix):

    normalized_inputs_ = normalized_inputs.clone()
    normalized_targets_ = normalized_targets.clone()

    # get the normalized projection predicitions of the model
    normalized_proj_predictions  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions), torch.tensor(normalized_inputs_), data_preprocessed, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions = torch.tensor(np.stack(normalized_proj_predictions))

    # compute the mape and sem with respect to target
    mape = mean_absolute_percentage_error(normalized_targets_, normalized_proj_predictions)
    rmse = np.sqrt(mean_squared_error(normalized_targets_, normalized_proj_predictions))

    return mape, rmse

# 
def select_random_rows(input_file, n, temp_file = 'data/ltp_system/temp.txt'):
    try:
        # Read all lines from the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Ensure N is not larger than the number of lines
        n = min(n, len(lines))
        
        # Randomly select N lines
        selected_lines = random.sample(lines, n)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        
        # Write the selected lines to the temp file
        with open(temp_file, 'w') as f:
            f.writelines(selected_lines)
        
        return temp_file
    except FileNotFoundError:
        return False, f"Error: The input file '{input_file}' was not found."
    except PermissionError:
        return False, f"Error: Permission denied when trying to create or write to '{temp_file}'."
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"

#
def split_dataset(input_file, n_testing_points, output_dir=None, testing_file=None, training_file=None):
    try:
        # Read all lines from the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Ensure n_testing_points is not larger than the total number of lines
        total_lines = len(lines)
        if n_testing_points >= total_lines:
            raise ValueError(f"n_testing_points ({n_testing_points}) must be less than the total number of lines in the dataset ({total_lines})")

        # Randomly select n_testing_points for the testing set
        testing_indices = set(random.sample(range(total_lines), n_testing_points))

        # Determine output file paths
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
        os.makedirs(output_dir, exist_ok=True)

        if testing_file is None:
            testing_file = os.path.join(output_dir, 'testing_data.txt')
        if training_file is None:
            training_file = os.path.join(output_dir, 'training_data.txt')

        # Write the data to testing and training files
        with open(testing_file, 'w') as test_f, open(training_file, 'w') as train_f:
            for i, line in enumerate(lines):
                if i in testing_indices:
                    test_f.write(line)
                else:
                    train_f.write(line)

        print(f"Dataset split successfully:")
        print(f"- Testing data ({n_testing_points} points) saved to: {testing_file}")
        print(f"- Training data ({total_lines - n_testing_points} points) saved to: {training_file}")

        # Return the paths of the created files
        return testing_file, training_file

    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied when trying to create or write to the output files.")
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    # Return None values if an error occurred
    return None, None

# get the overall RMSE statistics for each sample size across all random samples
def get_rmse_overall_statistics(n_samples_per_size, model_rmses, model_sigmas_rmse):
    """
    Compute overall RMSE and its uncertainty for a given dataset size.
    
    Args:
    - n_samples_per_size: Number of samples per dataset size.
    - model_rmses: List or array of RMSE values.
    - model_sigmas_rmse: List or array of RMSE uncertainties (if None, assumed to be zero).
    
    Returns:
    - rmse_overall: Mean RMSE over all samples.
    - sigma_rmse_overall: Combined uncertainty of the RMSE.
    """

    # convert lists to numpy arrays
    model_rmses = np.array(model_rmses)

    # the projection points do not account for dispersion among the bootstrapped model predictions
    if model_sigmas_rmse is None:
        model_sigmas_rmse = np.zeros_like(model_rmses)
    else:
        model_sigmas_rmse = np.array(model_sigmas_rmse)

    # rmse for each dataset size
    rmse_overall = np.mean(model_rmses)

    # uncertainty due to variability in RMSE values across samples
    sigma_rmse_samples = np.std(model_rmses, ddof=1) / np.sqrt(n_samples_per_size)

    # propagated uncertainty from individual predictions
    sigma_rmse_predictions = np.sqrt(np.mean(model_sigmas_rmse ** 2))

    # rmse uncertainty for each dataset size
    sigma_rmse_overall = np.sqrt(sigma_rmse_samples ** 2 + sigma_rmse_predictions ** 2)

    return rmse_overall, sigma_rmse_overall

# get the overall RMSE statistics for each sample size across all random samples
def get_mape_overall_statistics(n_samples_per_size, model_mapes, model_sigmas_mape):
    """
    Compute overall MAPE and its uncertainty for a given dataset size.
    
    Args:
    - n_samples_per_size: Number of samples per dataset size.
    - model_mapes: List or array of MAPE values.
    - model_sigmas_mape: List or array of MAPE uncertainties (if None, assumed to be zero).
    
    Returns:
    - mape_overall: Mean MAPE over all samples.
    - sigma_mape_overall: Combined uncertainty of the MAPE.
    """

    # convert lists to numpy arrays
    model_mapes = np.array(model_mapes)

    # the projection points do not account for dispersion among the bootstrapped model predictions
    if model_sigmas_mape is None:
        model_sigmas_mape = np.zeros_like(model_mapes)
    else:
        model_sigmas_mape = np.array(model_sigmas_mape)

    # rmse for each dataset size
    mape_overall = np.mean(model_mapes)

    # uncertainty due to variability in RMSE values across samples
    sigma_mape_samples = np.std(model_mapes, ddof=1) / np.sqrt(n_samples_per_size)

    # propagated uncertainty from individual predictions
    sigma_mape_predictions = np.sqrt(np.mean(model_sigmas_mape ** 2))

    # rmse uncertainty for each dataset size
    sigma_mape_overall = np.sqrt(sigma_mape_samples ** 2 + sigma_mape_predictions ** 2)

    return mape_overall, sigma_mape_overall

# main function to run the experiment
def run_experiment_6e(config_original, large_dataset_path, dataset_sizes, options):
    
    # setup the neural network architecture and the reference time for the dataset generation using LoKI
    hidden_sizes = options['hidden_sizes']
    activation_fns = options['activation_fns']
    loki_reference_time = config_original['dataset_generation']['loki_computation_time_100_points']
    config_ = generate_config_(config_original, hidden_sizes, activation_fns, options)
    
    # create the preprocessed_data object - all the smaller datasets should use the same scalers
    _, large_dataset = load_dataset(config_, large_dataset_path)
    data_preprocessing_info = DataPreprocessor(config_)
    data_preprocessing_info.setup_dataset(large_dataset.x, large_dataset.y)  

    # separate the test set of the large_dataset_path from the rest of the dataset which will be used to train the various models
    testing_file, training_file = split_dataset(large_dataset_path, n_testing_points = 300)
    test_inputs_norm, test_targets_norm = load_data(testing_file, data_preprocessing_info)

    # stores the results of the experiment
    results_file_path = 'output/ltp_system/checkpoints/different_datasets/results.csv'

    # number of different random samples for each dataset size
    n_samples_per_size = 30

    if options['RETRAIN_MODEL']:
        results = []

        # loop over the dataset sizes to generate
        for idx, dataset_size in enumerate(tqdm(dataset_sizes, desc="Evaluating Different Datasets")):
            
            nn_mapes, nn_sigmas_mape, nn_rmses, nn_sigmas_rmse = [], [], [], []
            proj_mapes, proj_rmses = [], []
            
            # for each dataset set create different samples
            for _ in range(n_samples_per_size):
            
                # 1. read from the train_path file and randomly select 'dataset_size' rows
                temp_file = select_random_rows(training_file, dataset_size)

                # 2. read the dataset and preprocess data
                _, temp_dataset = load_dataset(config_, temp_file)
                train_data_norm, val_data_norm = setup_dataset_with_preproprocessing_info(temp_dataset.x, temp_dataset.y, data_preprocessing_info)  

                # 4. create the val loader
                val_loader = torch.utils.data.DataLoader(val_data_norm, batch_size=config_['nn_model']['batch_size'], shuffle=True)

                # 5. train the neural network model (nn)
                nn_models, _, _, _ = get_trained_nn(config_, data_preprocessing_info, idx, train_data_norm, val_loader)

                # 6. use the trained nn to make predictions on the test inputs - get the normalized model predictions 
                test_inputs_norm_  = test_inputs_norm.clone() 
                test_targets_norm_ = test_targets_norm.clone() 
                nn_predictions_norm =  get_average_predictions(nn_models, torch.tensor(test_inputs_norm_))
                nn_pred_uncertainties =  get_predictive_uncertainty(nn_models, torch.tensor(test_inputs_norm_)) # for each point prediction gives an uncertainty value

                # 7. for the nn predictions compute the mape, rmse and mape_uncertainty
                nn_mape_j, nn_sigma_mape_j, nn_rmse_j, nn_sigma_rmse_j = evaluate_model(nn_predictions_norm, nn_pred_uncertainties, test_targets_norm_)
                nn_mapes.append(nn_mape_j)
                nn_rmses.append(nn_rmse_j)
                nn_sigmas_mape.append(nn_sigma_mape_j)
                nn_sigmas_rmse.append(nn_sigma_rmse_j)

                # 8. project the nn predictions and compute the mape, rmse and mape_uncertainty
                mape_proj, rmse_proj = evaluate_projection(nn_predictions_norm, test_targets_norm_, test_inputs_norm_, data_preprocessing_info, options['w_matrix'])
                proj_mapes.append(mape_proj)
                proj_rmses.append(rmse_proj)
            
            # 9. compute rmse and uncertainties for each dataset size
            nn_rmse_overall, nn_sigma_rmse_overall = get_rmse_overall_statistics(n_samples_per_size, nn_rmses, nn_sigmas_rmse)
            proj_rmse_overall, proj_sigma_rmse_overall = get_rmse_overall_statistics(n_samples_per_size, proj_rmses, model_sigmas_rmse = None)

            # 10. compute mape and uncertainties for each dataset size
            nn_mape_overall, nn_sigma_mape_overall = get_mape_overall_statistics(n_samples_per_size, nn_mapes, nn_sigmas_mape)
            proj_mape_overall, proj_sigma_mape_overall = get_mape_overall_statistics(n_samples_per_size, proj_mapes, model_sigmas_mape = None)

            # 11. compute the approximate time loki would take to generate that dataset
            loki_time = compute_approximate_loki_time(dataset_size, loki_reference_time)

            # 12. append the results 
            results.append((
                dataset_size, loki_time,
                nn_mape_overall, nn_sigma_mape_overall,      # MAPE statistics NN
                proj_mape_overall, proj_sigma_mape_overall,  # MAPE statistics PROJ
                nn_rmse_overall, nn_sigma_rmse_overall,      # RMSE statistics NN
                proj_rmse_overall, proj_sigma_rmse_overall   # RMSE statistics PROJ
                )
            )
        
        # store the final MAPE and RMSE results and corresponding uncertainties
        dataset_sizes, loki_times, nn_mapes, nn_mape_uncertainties, proj_mapes, proj_mape_uncertainties, nn_rmses, nn_rmse_uncertainties, proj_rmses, proj_rmse_uncertainties = zip(*results)
        data = {
            'dataset_sizes': dataset_sizes,
            'loki_time': loki_times,
            'nn_mapes': nn_mapes,
            'nn_mape_uncertainties': nn_mape_uncertainties,
            'proj_mapes': proj_mapes,
            'proj_mape_uncertainties': proj_mape_uncertainties,
            'nn_rmses': nn_rmses, 
            'nn_rmse_uncertainties': nn_rmse_uncertainties,
            'proj_rmses': proj_rmses,  
            'proj_rmse_uncertainties': proj_rmse_uncertainties
        }

        # Create a DataFrame from the results and store as .csv in a local directory
        df = pd.DataFrame(data)
        df = df.sort_values(by='dataset_sizes', ascending=True)
        df.to_csv(results_file_path, index=False)

        return df
    else:
        df = pd.read_csv(results_file_path)
        return df
    

# Plot the results
def Figure_6e(config_plotting, df):

    ############################################## MAPE PLOTS
    plt.figure(figsize=(10, 5))
    plt.xlabel('Dataset Size', fontsize=24)
    plt.ylabel('MAPE (\%)', fontsize=24)
    plt.errorbar(
        df['dataset_sizes'], df['nn_mapes'],                  # x, y
        yerr=df['nn_mape_uncertainties'],                     # y uncertainty
        fmt='-o', color=models_parameters['NN']['color'], 
        label='MAPE NN'
    )
    plt.errorbar(
        df['dataset_sizes'], df['proj_mapes'],                 # x, y
        yerr=df['proj_mape_uncertainties'],                    # y uncertainty
        fmt='-o', color=models_parameters['proj_nn']['color'], 
        linestyle='--', label='NN projection'
    )
    #plt.yscale('log') 
    plt.ylim(1e-2, 1e-1)  # Set y-axis range
    plt.tick_params(axis='y', labelsize=24)
    plt.tick_params(axis='x', labelsize=24) 
    plt.legend(loc='upper right', fontsize=24)
    # Save figures
    output_dir = config_plotting['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_6e_mape")
    savefig(save_path, pad_inches = 0.2)
    
    ############################################## RMSE PLOTS
    plt.figure(figsize=(10, 5))
    plt.xlabel('Dataset Size', fontsize=24)
    plt.ylabel('RMSE', fontsize=24)
    plt.errorbar(
        df['dataset_sizes'], df['nn_rmses'],              # x, y
        yerr=df['nn_rmse_uncertainties'],                 # y uncertainty 
        fmt='-o', color=models_parameters['NN']['color'], 
        label='RMSE NN'
    )
    plt.errorbar(
        df['dataset_sizes'], df['proj_rmses'], 
        yerr=df['proj_rmse_uncertainties'], 
        fmt='-o', color=models_parameters['proj_nn']['color'], 
        linestyle='--', label='NN projection'
    )
    #plt.yscale('log')
    plt.ylim(1e-3, 0.05)  # Set y-axis range
    plt.tick_params(axis='y', labelsize=24)
    plt.tick_params(axis='x', labelsize=24) 
    plt.legend(loc='upper right', fontsize=24)
    # Save figures
    output_dir = config_plotting['output_dir'] 
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_6e_rmse")
    savefig(save_path, pad_inches = 0.2)


