import os
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
from src.ltp_system.utils import set_seed, load_dataset, load_config, select_random_rows, split_dataset
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
def generate_config_(config, options):
    return {
        'dataset_generation' : config['dataset_generation'],
        'data_prep' : config['data_prep'],
        'nn_model': {
            'APPLY_EARLY_STOPPING': options['APPLY_EARLY_STOPPING'],
            'RETRAIN_MODEL'      : options['RETRAIN_MODEL'],
            'hidden_sizes'       : options['hidden_sizes'],
            'activation_fns'     : options['activation_fns'],
            'num_epochs'         : options['num_epochs'],
            'learning_rate'      : config['nn_model']['learning_rate'],
            'batch_size'         : config['nn_model']['batch_size'],
            'training_threshold' : config['nn_model']['training_threshold'],
            'n_bootstrap_models' : options['n_bootstrap_models'],
            'lambda_physics'     : config['nn_model']['lambda_physics'],       
            'patience'           : options['patience'],
            'alpha'              : options['alpha']     
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
def get_trained_nn(config, data_preprocessing_info, idx_dataset, idx_sample, train_data, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'different_datasets', f'dataset_{idx_dataset}_sample_{idx_sample}')
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
def evaluate_model(index_output_features, model_predictions_norm, model_pred_uncertainties, targets_norm):

    # perform a copy to avoid modifying the original arrays
    model_predictions_norm_ = (np.array(model_predictions_norm)).copy()
    model_pred_uncertainties_ = (np.array(model_pred_uncertainties)).copy()
    targets_norm_ = (np.array(targets_norm)).copy()

    m = len(model_pred_uncertainties)

    # compute the mape and the uncertainty 
    mape_all_outputs_j = np.mean(np.abs((targets_norm_ - model_predictions_norm_) / targets_norm_)) * 100
    mape_uncertainty_all_outputs_j = np.sqrt(np.mean(np.square(model_pred_uncertainties_ / targets_norm_))) * 100
    
    # compute the rmse and the uncertainty 
    rmse_all_outputs_j = np.sqrt(np.mean((targets_norm_ - model_predictions_norm_) ** 2))
    rmse_uncertainty_all_outputs_j = (1 / np.sqrt(m)) * np.sqrt(np.sum(np.square(model_pred_uncertainties_)))

    # perform a copy to avoid modifying the original arrays
    model_predictions_norm_ = (np.array(model_predictions_norm)).copy()
    model_pred_uncertainties_ = (np.array(model_pred_uncertainties)).copy()
    targets_norm_ = (np.array(targets_norm)).copy()

    # loop over the output features to compute the MAPE and RMSE for each output feature
    mapes_specific_outputs = [] # this should be a list: [mape_output_0, mape_output_1, ...]
    rmse_specific_outputs  = [] # this should be a list: [rmse_output_0, rmse_output_1, ...]
    if index_output_features is not None:
        
        # loop over each output given as argument to the test
        for i in index_output_features:
            model_predictions_norm_output_i   = model_predictions_norm_[:, i]
            target_norm_output_i = targets_norm_[:, i]

            # compute the MAPE of the i-th output feature with respect to target
            mape_output_i = np.mean(np.abs((target_norm_output_i - model_predictions_norm_output_i) / target_norm_output_i)) * 100

            # compute the RMSE of the i-th output feature with respect to target
            rmse_output_i = np.sqrt(np.mean((target_norm_output_i - model_predictions_norm_output_i) ** 2))
            
            # append results
            mapes_specific_outputs.append(mape_output_i)
            rmse_specific_outputs.append(rmse_output_i)

    # nn_mape_j, nn_sigma_mape_j, nn_rmse_j, nn_sigma_rmse_j, specific_outputs_mapes_nn_j, specific_outputs_rmses_nn_j
    return mape_all_outputs_j, mape_uncertainty_all_outputs_j, rmse_all_outputs_j, rmse_uncertainty_all_outputs_j, mapes_specific_outputs, rmse_specific_outputs

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

    # compute the mape and the rmse 
    mape_all_outputs_j = np.mean(np.abs((normalized_targets_ - normalized_proj_predictions_) / normalized_targets_)) * 100
    rmse_all_outputs_j = np.sqrt(np.mean((normalized_targets_ - normalized_proj_predictions_) ** 2))

    # perform a copy to avoid modifying the original arrays
    normalized_proj_predictions_ = (normalized_proj_predictions).copy()
    normalized_targets_ = (np.array(normalized_targets)).copy()

    # loop over the output features to compute the MAPE and RMSE for each output feature
    mapes_specific_outputs = [] # this should be a list: [mape_output_0, mape_output_1, ...]
    rmse_specific_outputs  = [] # this should be a list: [rmse_output_0, rmse_output_1, ...]

    if index_output_features is not None:
        for i in index_output_features:
            # extract predictions for each of the individual outputs of interest
            normalized_preds_i   = normalized_proj_predictions_[:, i]
            normalized_targets_i = normalized_targets_[:, i]

            # compute the MAPE of the i-th output feature with respect to target
            mape_output_i = np.mean(np.abs((normalized_targets_i - normalized_preds_i) / normalized_targets_i)) * 100

            # compute the RMSE of the i-th output feature with respect to target
            rmse_output_i = np.sqrt(np.mean((normalized_targets_i - normalized_preds_i) ** 2))

            mapes_specific_outputs.append(mape_output_i)
            rmse_specific_outputs.append(rmse_output_i)
    
    #proj_mape_j, proj_rmse_j, specific_outputs_proj_mapes_j, specific_outputs_proj_rmses_j
    return mape_all_outputs_j, rmse_all_outputs_j, mapes_specific_outputs, rmse_specific_outputs

# get the overall RMSE statistics for each sample size across all random samples
"""def get_rmse_overall_statistics(n_samples_per_size, model_rmses, model_sigmas_rmse):

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
    #sigma_rmse_predictions = np.sqrt(np.mean(model_sigmas_rmse ** 2))

    # rmse uncertainty for each dataset size
    #sigma_rmse_overall = np.sqrt(sigma_rmse_samples ** 2 + sigma_rmse_predictions ** 2)

    return rmse_overall, sigma_rmse_samples"""

"""# get the overall RMSE statistics for each sample size across all random samples
def get_mape_overall_statistics(n_samples_per_size, model_mapes, model_sigmas_mape):

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
    #sigma_mape_predictions = np.sqrt(np.mean(model_sigmas_mape ** 2))

    # rmse uncertainty for each dataset size
    #sigma_mape_overall = np.sqrt(sigma_mape_samples ** 2 + sigma_mape_predictions ** 2)

    return mape_overall, sigma_mape_samples"""

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
def run_experiment_6e(config_original, large_dataset_path, dataset_sizes, options, n_testing_points = 500):

    ###################################### 1. SETUP AND DEFINITIONS ###################################
    # get the index of the output features to extract
    output_features = config_original['dataset_generation']['output_features']
    index_output_features = [output_features.index(feature) for feature in options['extract_results_specific_outputs']]

    # setup the neural network architecture and the reference time for the dataset generation using LoKI
    loki_reference_time = config_original['dataset_generation']['loki_computation_time_100_points']
    config_ = generate_config_(config_original, options)

    # define the file paths for the results
    table_dir = os.path.join(options['output_dir'], 'table_results')
    os.makedirs(table_dir, exist_ok=True) # create the directory if it doesn't exist
    all_results_file_path = os.path.join(table_dir, 'all_outputs_mean_results.csv')
    specific_outputs_file_path = os.path.join(table_dir, 'specific_outputs_results.csv')

    # number of different random samples for each dataset size
    n_samples_per_size = options['n_samples']
    ###################################################################################################

    ###################################### 2. DEAL WITH DATA SELECTION ################################
    data_preprocessing_info, training_file, test_inputs_norm, test_targets_norm = split_dataset_(config_, large_dataset_path, n_testing_points)
    # Save the data_preprocessing_info object
    directory = "output/ltp_system/checkpoints/different_datasets"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "data_preprocessing_info.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(data_preprocessing_info, file)
    ###################################################################################################


    ###################################### 3. GET RESULTS FOR EACH DATASET SIZE #######################
    # if the model is being retrained, compute the results for each dataset size
    if options['RETRAIN_MODEL']:
        all_outputs_results = []
        specific_outputs_rows = []

        # loop over the dataset sizes to generate
        for idx_dataset, dataset_size in enumerate(tqdm(dataset_sizes, desc="Evaluating Different Datasets")):
            
            # initialize the results for the mean of all outputs
            nn_mapes, nn_sigmas_mape, nn_rmses, nn_sigmas_rmse, proj_mapes, proj_rmses = [], [], [], [], [], []
            
            # initialize the results for the specific outputs
            specific_outputs_mapes_nn, specific_outputs_rmses_nn, specific_outputs_mapes_proj, specific_outputs_rmses_proj = [], [], [], []
            
            # for each dataset set create different samples
            for sample_i in range(n_samples_per_size):
            
                # 1. read from the train_path file and randomly select 'dataset_size' rows - save the dataset to a local dir sampled_dataset_dir
                sampled_dataset_dir = select_random_rows(training_file, dataset_size, seed = sample_i + 1)

                # 2. read the dataset from the sampled_dataset_dir and preprocess data
                _, sampled_dataset = load_dataset(config_, sampled_dataset_dir)

                # 3. preprocess the data: the training sets, ie, subsets of the bigger dataset, are preprocessed using the scalers fitted on the large dataset to avoid data leakage.
                train_data_norm, _, val_data_norm  = setup_dataset_with_preproprocessing_info(sampled_dataset.x, sampled_dataset.y, data_preprocessing_info)  

                # 4. create the val loader needed for training the NN.
                val_loader = torch.utils.data.DataLoader(val_data_norm, batch_size=config_['nn_model']['batch_size'], shuffle=True)

                # 5. train the neural network model (nn) on the sampled training data
                nn_models, _, _, _ = get_trained_nn(config_, data_preprocessing_info, idx_dataset, sample_i, train_data_norm, val_loader)

                # 6. perform copies of the test inputs and test targets to avoid modifying them.
                test_inputs_norm_  = test_inputs_norm.clone() 
                test_targets_norm_ = test_targets_norm.clone() 

                # 7. use the trained nn to make predictions on the test inputs - get the normalized model predictions 
                nn_predictions_norm =  get_average_predictions(nn_models, torch.tensor(test_inputs_norm_))
                nn_pred_uncertainties =  get_predictive_uncertainty(nn_models, torch.tensor(test_inputs_norm_)) # for each point prediction gives an uncertainty value
                
                # 8. perform copies of the test inputs and test targets to avoid modifying them.
                nn_predictions_norm_  = nn_predictions_norm.clone() 
                nn_pred_uncertainties_ = nn_pred_uncertainties.clone() 
                test_targets_norm_ = test_targets_norm.clone() 

                # 9. for the nn predictions compute the mape, rmse and uncertainties (sigma/sqrt(n))
                nn_mape_j, nn_sigma_mape_j, nn_rmse_j, nn_sigma_rmse_j, specific_outputs_mapes_nn_j, specific_outputs_rmses_nn_j = evaluate_model(index_output_features, nn_predictions_norm_, nn_pred_uncertainties_, test_targets_norm_)
                nn_mapes.append(nn_mape_j)
                nn_rmses.append(nn_rmse_j)
                nn_sigmas_mape.append(nn_sigma_mape_j)
                nn_sigmas_rmse.append(nn_sigma_rmse_j)
                specific_outputs_mapes_nn.append(specific_outputs_mapes_nn_j)
                specific_outputs_rmses_nn.append(specific_outputs_rmses_nn_j)

                # 10. project the nn predictions and compute the mape, rmse and uncertainties (sigma/sqrt(n))
                proj_mape_j, proj_rmse_j, specific_outputs_proj_mapes_j, specific_outputs_proj_rmses_j = evaluate_projection(index_output_features, nn_predictions_norm, test_targets_norm_, test_inputs_norm_, data_preprocessing_info, options['w_matrix'])
                proj_mapes.append(proj_mape_j)
                proj_rmses.append(proj_rmse_j)
                specific_outputs_mapes_proj.append(specific_outputs_proj_mapes_j)
                specific_outputs_rmses_proj.append(specific_outputs_proj_rmses_j)
            
            # 11. compute mean of all outputs rmse and mape
            nn_rmse_overall   = np.mean(nn_rmses)
            nn_mape_overall   = np.mean(nn_mapes)
            proj_rmse_overall = np.mean(proj_rmses)
            proj_mape_overall = np.mean(proj_mapes)
            
            # 12. compute uncertainties of the errors
            nn_sigma_rmse_overall = np.std(nn_rmses, ddof=1) / np.sqrt(n_samples_per_size)
            nn_sigma_mape_overall = np.std(nn_mapes, ddof=1) / np.sqrt(n_samples_per_size)
            proj_sigma_rmse_overall = np.std(proj_rmses, ddof=1) / np.sqrt(n_samples_per_size)
            proj_sigma_mape_overall = np.std(proj_mapes, ddof=1) / np.sqrt(n_samples_per_size)

            # 13. compute the approximate time loki would take to generate that dataset
            loki_time = compute_approximate_loki_time(dataset_size, loki_reference_time)
            
            # 14. append the results 
            all_outputs_results.append((
                dataset_size, loki_time,
                nn_mape_overall, nn_sigma_mape_overall,      # MAPE statistics NN
                proj_mape_overall, proj_sigma_mape_overall,  # MAPE statistics PROJ
                nn_rmse_overall, nn_sigma_rmse_overall,      # RMSE statistics NN
                proj_rmse_overall, proj_sigma_rmse_overall   # RMSE statistics PROJ
                )
            )
            
            # /// CREATE THE DATAFRAME FOR THE RESULTS CONCERNING THE SPECIFIC OUTPUTS ///
            if options['extract_results_specific_outputs'] is not None:
                specific_outputs_rmses_proj = np.array(specific_outputs_rmses_proj)
                specific_outputs_mapes_proj = np.array(specific_outputs_mapes_proj) 
                specific_outputs_rmses_nn = np.array(specific_outputs_rmses_nn)
                specific_outputs_mapes_nn = np.array(specific_outputs_mapes_nn)
                
                # 15. compute specific outputs RMSE and MAPE across all samples
                for output_idx, output_feature in enumerate(index_output_features):

                    # get the specific outputs errors for the current output
                    nn_rmses_output_i = specific_outputs_rmses_nn[:,output_idx]       # for each output: [nn_rmse_sample_0, nn_rmse_sample_1, ...]
                    nn_mapes_output_i = specific_outputs_mapes_nn[:,output_idx]       # for each output: [nn_mape_sample_0, nn_mape_sample_1, ...]
                    proj_rmses_output_i = specific_outputs_rmses_proj[:,output_idx]   # for each output: [proj_rmse_sample_0, proj_rmse_sample_1, ...]
                    proj_mapes_output_i = specific_outputs_mapes_proj[:,output_idx]   # for each output: [proj_mape_sample_0, proj_mape_sample_1, ...]

                    # compute the mean of the specific outputs errors across all samples # nn_rmses_output_i, nn_mapes_output_i, proj_rmses_output_i, proj_mapes_output_i
                    specific_outputs_rmses_nn_overall   = np.mean(nn_rmses_output_i)
                    specific_outputs_mapes_nn_overall   = np.mean(nn_mapes_output_i)
                    specific_outputs_rmses_proj_overall = np.mean(proj_rmses_output_i)
                    specific_outputs_mapes_proj_overall = np.mean(proj_mapes_output_i)

                    # append the results for the current output
                    row = [dataset_size, loki_time, output_feature, specific_outputs_mapes_nn_overall, specific_outputs_mapes_proj_overall, specific_outputs_rmses_nn_overall, specific_outputs_rmses_proj_overall]
                    specific_outputs_rows.append(row)
        
        # STORE THE RESULTS FOR THE MEAN OF ALL OUTPUTS #########################################################
        dataset_sizes, loki_times, nn_mapes, nn_mape_uncertainties, proj_mapes, proj_mape_uncertainties, nn_rmses, nn_rmse_uncertainties, proj_rmses, proj_rmse_uncertainties = zip(*all_outputs_results)
        data_all_outputs = {
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
        df_all_outputs = pd.DataFrame(data_all_outputs)
        df_all_outputs = df_all_outputs.sort_values(by='dataset_sizes', ascending=True)
        df_all_outputs.to_csv(all_results_file_path, index=False)
        
        # STORE THE RESULTS FOR THE SPECIFIC OUTPUTS
        if options['extract_results_specific_outputs'] is not None:
            cols_names = ['dataset_sizes', 'loki_time', 'output_feature', 'nn_mapes', 'proj_mapes', 'nn_rmses', 'proj_rmses']
            df_specific_outputs = pd.DataFrame(specific_outputs_rows, columns = cols_names)
            df_specific_outputs = df_specific_outputs.sort_values(by='dataset_sizes', ascending=True)
            df_specific_outputs.to_csv(specific_outputs_file_path, index=False)
        
        # return results
        return df_all_outputs, df_specific_outputs, data_preprocessing_info
    
    else:
        df_all_outputs = pd.read_csv(all_results_file_path)
        df_specific_outputs = pd.read_csv(specific_outputs_file_path)
        
        return df_all_outputs, df_specific_outputs, data_preprocessing_info
    
# Plot the results for the mean of all outputs
def Figure_6e_mean_all_outputs(options, df):

    # Plot MAPE for NN and NN projection
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel('Dataset Size', fontsize=24)
    ax1.set_ylabel('MAPE (\%)', fontsize=24)
    ax1.plot(df['dataset_sizes'], df['nn_mapes'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['dataset_sizes'], df['proj_mapes'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24)
    ax1.tick_params(axis='x', labelsize=24)
    ax1.legend(loc='upper right', fontsize=20)
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAPE Variation Rate (\%)', fontsize=24, color='gray')
    improvement_rate = (df['proj_mapes'] - df['nn_mapes']) / df['nn_mapes'] * 100
    ax2.plot(df['dataset_sizes'], improvement_rate, '-o', color='gray', label='Improvement Rate (%)') 
    ax2.axhline(0, color='lightgray', linestyle='--', linewidth=2)  # Horizontal line at y=0
    ax2.tick_params(axis='y', labelsize=24, colors='gray')  # Set tick color to gray
    ax2.spines['right'].set_color('gray')  # Set color of the second y-axis spine to gray
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
    ax1.set_xlabel('Dataset Size', fontsize=24)
    ax1.set_ylabel('RMSE', fontsize=24, labelpad=10)
    ax1.set_yscale('log')
    ax1.set_ylim(5e-2, 30e-2)
    ax1.set_yticks([5e-2, 1e-1, 2e-1])
    ax1.set_yticklabels(['5', '10', '20'])
        # Plot lines without error bars
    ax1.plot(df['dataset_sizes'], df['nn_rmses'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['dataset_sizes'], df['proj_rmses'],'-o', color=models_parameters['proj_nn']['color'],linestyle='--', label='NN projection')
    # Add error bars to the plots
    #ax1.errorbar(df['dataset_sizes'], df['nn_rmses'], yerr=df['nn_rmse_uncertainties'],fmt='-o', color=models_parameters['NN']['color'], label='NN', capsize=5)
    #ax1.errorbar(df['dataset_sizes'], df['proj_rmses'], yerr=df['proj_rmse_uncertainties'],fmt='-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection', capsize=5)
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
    improvement_rate_rmse = (df['proj_rmses'] - df['nn_rmses']) / df['nn_rmses'] * 100
    ax2.plot(df['dataset_sizes'], improvement_rate_rmse, '-o', color='gray', label='Improvement Rate (%)')  # Set line color to gray
    ax2.tick_params(axis='y', labelsize=24, colors='gray')  # Set tick color to gray
    ax2.spines['right'].set_color('gray')  # Set color of the second y-axis spine to gray
    #fig.legend(loc='right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    # Save figure
    fig.tight_layout()
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_all_outputs_rmse.pdf")
    fig.savefig(save_path, pad_inches=0.2, format='pdf', dpi=300, bbox_inches='tight')

# Plot the results for the specific outputs
def Figure_6e_specific_outputs(options, df_all, df_specific, output_features_names):
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
        ax.set_xlabel('Dataset Size', fontsize=24, fontweight='bold')
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
        line1, = ax.plot(df_specific_output['dataset_sizes'], df_specific_output['nn_mapes'], '-o', color=models_parameters['NN']['color'],linewidth=3, markersize=10)
        line2, = ax.plot(df_specific_output['dataset_sizes'], df_specific_output['proj_mapes'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--',linewidth=3, markersize=10)
        
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
    output_dir = options['output_dir'] + "plots"
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
        line1, = ax.plot(df_specific_output['dataset_sizes'], df_specific_output['nn_rmses'], '-o', color=models_parameters['NN']['color'],linewidth=3, markersize=10)
        line2, = ax.plot(df_specific_output['dataset_sizes'], df_specific_output['proj_rmses'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--',linewidth=3, markersize=10)
        
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
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "specific_outputs_rmse.pdf")
    fig.savefig(save_path, pad_inches=0.3, format='pdf', dpi=300, bbox_inches='tight')

