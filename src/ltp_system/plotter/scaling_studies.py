import io
import os
import csv
import pickle
import torch
import random
import contextlib
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Tuple
from contextlib import redirect_stdout, redirect_stderr

from src.ltp_system.utils import savefig, set_seed, load_dataset, select_random_rows, sample_dataset
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

#///////////////////////////////////////////////////////////////////////////////////////#
#////////////////////// COMMON FUNCTIONS TO BOTH STUDIES ///////////////////////////////#
#///////////////////////////////////////////////////////////////////////////////////////#

# create a configuration file for the chosen architecture
def generate_config_(config, hidden_sizes, options):
    return {
        'dataset_generation' : config['dataset_generation'],
        'data_prep' : config['data_prep'],
        'nn_model': {
            'APPLY_EARLY_STOPPING': options['APPLY_EARLY_STOPPING'],
            'RETRAIN_MODEL'      : options['RETRAIN_MODEL'],
            'hidden_sizes'       : options['hidden_sizes'] if hidden_sizes is None else hidden_sizes,
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

# train the nn for a chosen architecture or load the parameters if it has been trained 
def get_trained_nn(options, config, data_preprocessing_info, idx_dataset, idx_sample, train_data, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join(options['checkpoints_dir'], f'dataset_{idx_dataset}_sample_{idx_sample}') 
    os.makedirs(checkpoint_dir, exist_ok=True)

    if options['RETRAIN_MODEL']:
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                nn_models, nn_losses_dict, training_time = get_trained_bootstraped_models(config['nn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')

        return nn_models, nn_losses_dict, device, training_time
    else:
        try:
            nn_models, _, hidden_sizes, activation_fns, training_time = load_checkpoints(config['nn_model'], NeuralNetwork, checkpoint_dir)
            return nn_models, hidden_sizes, activation_fns, training_time
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")

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
    data_preprocessing_info.setup_dataset(large_dataset.x, large_dataset.y, print_messages=False)  

    # separate the test set of the large_dataset_path from the rest of the dataset which will be used to train the various models
    testing_file, training_file = sample_dataset(large_dataset_path, n_testing_points)
    test_inputs_norm, test_targets_norm = load_data(testing_file, data_preprocessing_info)

    return data_preprocessing_info, training_file, test_inputs_norm, test_targets_norm

# main function to run the experiment
def run_experiment(config_original, large_dataset_path, dataset_sizes, options, n_testing_points = 500):

    ###################################### 1. SETUP AND DEFINITIONS ###################################
    set_seed(42)
    # get the index of the output features to extract
    output_features = config_original['dataset_generation']['output_features']
    index_output_features = [output_features.index(feature) for feature in options['extract_results_specific_outputs']]
    config_ = generate_config_(config_original, None, options)

    # define the file paths for the results
    table_dir = os.path.join(options['output_dir'], 'table_results')
    os.makedirs(table_dir, exist_ok=True) # create the directory if it doesn't exist
    all_results_file_path = os.path.join(table_dir, 'all_outputs_mean_results.csv')
    specific_outputs_file_path = os.path.join(table_dir, 'specific_outputs_results.csv')
    n_samples_per_size = options['n_samples'] # number of different random samples for each dataset size
    ###################################################################################################

    ###################################### 2. DEAL WITH DATA SELECTION ################################
    data_preprocessing_info, training_file, test_inputs_norm, test_targets_norm = split_dataset_(config_, large_dataset_path, n_testing_points)
    
    # Save the data_preprocessing_info object
    os.makedirs(options['checkpoints_dir'], exist_ok=True)
    file_path = os.path.join(options['checkpoints_dir'], "data_preprocessing_info.pkl")
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
                sampled_dataset_dir = select_random_rows(training_file, dataset_size, seed = sample_i + 1, print_messages=False)

                # 2. read the dataset from the sampled_dataset_dir and preprocess data
                _, sampled_dataset = load_dataset(config_, sampled_dataset_dir)

                # 3. preprocess the data: the training sets, ie, subsets of the bigger dataset, are preprocessed using the scalers fitted on the large dataset to avoid data leakage.
                train_data_norm, _, val_data_norm  = setup_dataset_with_preproprocessing_info(sampled_dataset.x, sampled_dataset.y, data_preprocessing_info, print_messages = False)  

                # 4. create the val loader needed for training the NN.
                val_loader = torch.utils.data.DataLoader(val_data_norm, batch_size=config_['nn_model']['batch_size'], shuffle=True)

                # 5. train the neural network model (nn) on the sampled training data
                nn_models, _, _, _ = get_trained_nn(options, config_, data_preprocessing_info, idx_dataset, sample_i, train_data_norm, val_loader)

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
            
            # 13. append the results 
            all_outputs_results.append((
                dataset_size,
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
                    row = [dataset_size, output_feature, specific_outputs_mapes_nn_overall, specific_outputs_mapes_proj_overall, specific_outputs_rmses_nn_overall, specific_outputs_rmses_proj_overall]
                    specific_outputs_rows.append(row)
        
        # STORE THE RESULTS FOR THE MEAN OF ALL OUTPUTS #########################################################
        dataset_sizes, nn_mapes, nn_mape_uncertainties, proj_mapes, proj_mape_uncertainties, nn_rmses, nn_rmse_uncertainties, proj_rmses, proj_rmse_uncertainties = zip(*all_outputs_results)
        data_all_outputs = {
            'dataset_sizes': dataset_sizes,
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
            cols_names = ['dataset_sizes', 'output_feature', 'nn_mapes', 'proj_mapes', 'nn_rmses', 'proj_rmses']
            df_specific_outputs = pd.DataFrame(specific_outputs_rows, columns = cols_names)
            df_specific_outputs = df_specific_outputs.sort_values(by='dataset_sizes', ascending=True)
            df_specific_outputs.to_csv(specific_outputs_file_path, index=False)
        
        # return results
        return df_all_outputs, df_specific_outputs
    
    else:
        df_all_outputs = pd.read_csv(all_results_file_path)
        df_specific_outputs = pd.read_csv(specific_outputs_file_path)
        
        return df_all_outputs, df_specific_outputs


#///////////////////////////////////////////////////////////////////////////////////////#
#///////////////////////////// FUNCTIONS USED FOR PLOTTING  ////////////////////////////#
#///////////////////////////////////////////////////////////////////////////////////////#
# used for defining the y-axis boundaries in the plots
def get_min_max_bounds(df_specific, metric, log_scale):
    # For proj_mapes
    proj_min = df_specific['proj_' + metric].min()
    proj_max = df_specific['proj_' + metric].max()

    # For nn_mapes
    nn_min = df_specific['nn_' + metric].min()
    nn_max = df_specific['nn_' + metric].max()

    # Or to get the overall min and max across both columns
    overall_min = min(proj_min, nn_min)
    overall_max = max(proj_max, nn_max)

    if log_scale:
        # For linear scale (e.g., MAPE), use additive buffer
        initial_buffer = (overall_max - overall_min) * 0.1
        buffer = initial_buffer

        # Reduce buffer if y_min would be negative or zero
        while overall_min - buffer <= 0:
            buffer *= 0.5  # Reduce buffer by half each time

        y_min = overall_min - buffer
        y_max = overall_max + buffer

    else:
        # For log scale, we use multiplicative buffer instead of additive
        buffer_factor = 1.1  # 10% buffer
        y_min = overall_min / buffer_factor
        y_max = overall_max * buffer_factor

    return y_min, y_max

# used in the specific outputs plots as a function of dataset size or number of parameters
def _create_subfigures(fig, axes, options_plot, df_specific, n_cols, output_features_names, specific_outputs, labels, analysis, metric):

    axes = axes.flatten()

    for plot_idx, output_idx in enumerate(df_specific['output_feature'].unique()):
        
        # Extract dataframe with data for specific output
        ax = axes[plot_idx]
        df_specific_output = df_specific[df_specific['output_feature'] == output_idx]
        
        # Format x and y axis: axis labels (only add ylabel for first plot in each row)
        ax.set_xlabel(options_plot['x_axis']['label'], fontsize=24, labelpad=10)
        if plot_idx % n_cols == 0:  
            ax.set_ylabel(options_plot['y_axis']['label'], fontsize=24, fontweight='bold', labelpad=15)
        else:
            ax.set_ylabel('')

        # Format x and y axis: log scale
        if options_plot['y_axis']['log_scale']:
            ax.set_yscale('log')

        # Format x and y axis: log scale
        if options_plot['x_axis']['log_scale']:
            ax.set_xscale('log')
        
        # Format plot frame
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # Plot the RMSE of the NN and the NN projection as a function of the dataset size 
        line1, = ax.plot(df_specific_output[analysis], df_specific_output[f'nn_{metric}'], '-o', color=models_parameters['NN']['color'],linewidth=3, markersize=10)
        line2, = ax.plot(df_specific_output[analysis], df_specific_output[f'proj_{metric}'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--',linewidth=3, markersize=10)
        
        # Format x and y axis: ticks style
        ax.tick_params(axis='both', which='major', labelsize=24, width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=2, length=4)

        # Format title
        ax.set_title(f'{output_features_names[output_idx]}', fontsize=28, pad=15)
        
        # Format y axis: ticks and y range
        current_output = specific_outputs[plot_idx]
        ax.set_ylim(options_plot[current_output]['y_min'], options_plot[current_output]['y_max']) 
        ax.set_yticks(options_plot[current_output]['y_ticks'])
        ax.set_yticklabels(options_plot[current_output]['y_tick_labels'])
        if(options_plot[current_output]['y_label_position'] is not None):
            ax.text(0, options_plot[current_output]['y_label_position'], options_plot[current_output]['y_over_axis_label'], transform=ax.get_yaxis_transform(), fontsize=24)
        ax.minorticks_off()

        if(analysis == 'num_params' and metric == 'rmses'):
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'$10^{int(np.log10(x))}$'))

    # Remove any extra subplots
    for idx in range(plot_idx+1, len(axes)):
        fig.delaxes(axes[idx])

    # Get lines and labels
    lines = [line1, line2]

    # Add single legend at the top
    fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.36, 0.34), ncol=3, fontsize=24, frameon=True)

    return fig

# main function for specific outputs plots as a function of dataset size or number of parameters
def create_specific_output_plot(options, options_plot_mape, options_plot_rmse,  df_specific, output_features_names, analysis):

    # Create a grid of subplots for specific outputs
    n_outputs = len(df_specific['output_feature'].unique())
    n_cols = 3
    n_rows = (n_outputs + 1) // 2  # Ceiling division to handle odd number of plots

    # Get the names of the specific outputs to plot
    labels = ['NN', 'NN projection']
    specific_outputs = options['extract_results_specific_outputs']

    for options_plot_err, metric in zip([options_plot_mape, options_plot_rmse], ['mapes', 'rmses']):
        # Clear current plt and axes
        plt.clf()
        plt.cla()

        # Initialize the figure 
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5*n_rows))

        # Create the plots for the error metric where each subplot is a specific output
        fig = _create_subfigures(fig, axes, options_plot_err, df_specific, n_cols, output_features_names, specific_outputs, labels, analysis, metric)

        # Adjust layout and save
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Make room for legend
        output_dir = options['output_dir'] + "plots"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"specific_outputs_{metric}.pdf")
        fig.savefig(save_path, pad_inches=0.3, format='pdf', dpi=300, bbox_inches='tight')

    print(f"\nResults of RMSE of specific outputs ({specific_outputs}) saved as .pdf file to:\n   → {output_dir}.")


#///////////////////////////////////////////////////////////////////////////////////////#
#///////////////////// ABLATION STUDY (different architectures)/////////////////////////#
#///////////////////////////////////////////////////////////////////////////////////////#
 
# Compute the number of weights and biases in the nn
def compute_parameters(layer_config):

    layer_config = [3] + layer_config + [17]

    hidden_layers = layer_config[1:-1]
    n_weights = sum(layer_config[i] * layer_config[i+1] for i in range(len(layer_config) - 1)) # x[layer_0] * x[layer_1] + ... + x[layer_N-1]*x[layer_N]
    n_biases = sum(hidden_layers)

    return n_weights + n_biases

# Returns list of sublists with the architectures to analyse
def generate_random_architectures(options):
    
    min_neurons_per_layer = options['min_neurons_per_layer']
    max_neurons_per_layer = options['max_neurons_per_layer']
    n_architectures = options['n_steps']
    n_hidden_layers = options['n_hidden_layers']

    print(f"\nGenerating {n_architectures} NN architectures ...")

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
    architectures = np.unique(np.sort(architectures))  

    # Convert each element to a sublist of repeated neurons
    architectures_list = []
    for architecture in architectures:
        sublist = [architecture] * n_hidden_layers
        architectures_list.append(sublist)
        
    return architectures_list

# Either generates the list of architectures or retrieves the list from a local directory
def get_random_architectures(options, architectures_file_path):
    if options['RETRAIN_MODEL']:
        # generate random nn architectures
        random_architectures_list = generate_random_architectures(options)
        # save the nn architectures to a file
        with open(architectures_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(random_architectures_list)
        print(f"Saved NN architectures as .csv to:\n   → {architectures_file_path}")
    else:
        # load the nn architectures from a file
        random_architectures_list = []
        with open(architectures_file_path, mode='r') as file:
            reader = csv.reader(file)
            random_architectures_list = [row for row in reader]
        random_architectures_list = [[int(num) for num in sublist] for sublist in random_architectures_list]

        print(f"Loaded NN architectures from:\n   → {architectures_file_path}")
    
    return random_architectures_list

# Main function to run the experiment
def run_ablation_study_architectures(config_original, large_dataset_path, options):
    print("\n\n")
    print("=" * 20 + " Ablation Study for Different Architectures " + "=" * 20)

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

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                df_results_all_architecture_idx, df_results_specific_architecture_idx = run_experiment(
                    config_, 
                    large_dataset_path, 
                    dataset_size, 
                    options
                )
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
        cols_names = ['architectures', 'num_params', 'nn_mapes', 'uncertanties_mape_nn', 'proj_mapes', 'uncertanties_mape_proj', 'nn_rmses', ' uncertanties_rmse_nn', 'proj_rmses', 'uncertanties_rmse_proj']
        df_all_outputs = pd.DataFrame(rows_mean_all_outputs, columns = cols_names)
        df_all_outputs = df_all_outputs.sort_values(by='num_params', ascending=True)
        df_all_outputs.to_csv(all_results_file_path, index=False)
        print(f"\nResults of mean RMSE across all outputs saved as .csv files to:\n   → {all_results_file_path}.")

        # create dataframe with mean of all outputs results
        cols_names = ['architecture', 'num_params', 'output_feature', 'nn_mapes', 'proj_mapes', 'nn_rmses', 'proj_rmses']
        df_specific_outputs = pd.DataFrame(rows_specific_outputs, columns = cols_names)
        df_specific_outputs = df_specific_outputs.sort_values(by='num_params', ascending=True)
        df_specific_outputs.to_csv(specific_outputs_file_path, index=False)
        print(f"\nResults of RMSE of specific outputs ({index_output_features}) saved as .csv files to:\n   → {specific_outputs_file_path}.")


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
    ax1.set_ylabel('MAPE (\%)', fontsize=24, fontweight='bold', labelpad=15)
    ax1.set_xscale('log')
    ax1.plot(df['num_params'], df['nn_mapes'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['proj_mapes'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24, width = 2, length = 6)
    ax1.tick_params(axis='x', labelsize=24, width = 2, length = 6)
    #ax1.legend(loc='center right', fontsize=20)
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAPE Variation Rate (\%)', fontsize=24, color='#B8B42D')
    ax2.set_xscale('log')
    improvement_rate = (df['proj_mapes'] - df['nn_mapes']) / df['nn_mapes'] * 100
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
    ax1.set_ylabel('RMSE', fontsize=24, fontweight='bold', labelpad=15)
    #ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylim(0, 28e-2) # ax1.set_ylim(2e-2, 30e-2)
    ax1.set_yticks([5e-2, 10e-2, 15e-2, 20e-2, 25e-2])    # ax1.set_yticks([3e-2, 5e-2, 1e-1, 2e-1])
    ax1.set_yticklabels(['5', '10', '15', '20', '25']) # ax1.set_yticklabels(['3', '5', '10', '20'])
        # Plot lines without error bars
    ax1.plot(df['num_params'], df['nn_rmses'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['num_params'], df['proj_rmses'],'-o', color=models_parameters['proj_nn']['color'],linestyle='--', label='NN projection')
    # Add error bars to the plots
    #ax1.errorbar(df['num_params'], df['rmses_nn'], yerr=df['uncertanties_rmse_nn'],fmt='-o', color=models_parameters['NN']['color'], label='NN', capsize=5)
    #ax1.errorbar(df['num_params'], df['rmses_proj'], yerr=df['uncertanties_rmse_proj'],fmt='-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection', capsize=5)
    ax1.tick_params(axis='y', labelsize=24, width = 2, length = 6)
    ax1.tick_params(axis='x', labelsize=24, width = 2, length = 6)
    #ax1.legend(loc='right', fontsize=20)
    # Add scientific notation label at the top of the y-axis
    plt.minorticks_off()
    y_max = plt.gca().get_ylim()[1] * 3.62
    x_min = plt.gca().get_ylim()[0] #- 0.05
    ax1.text(x_min, 1.08, r'($\times10^{-2}$)', transform=plt.gca().transAxes, fontsize=24, ha='left', va='center')
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE Variation Rate (\%)', fontsize=24, color='#B8B42D')
    ax2.set_xscale('log')
    improvement_rate_rmse = (df['proj_rmses'] - df['nn_rmses']) / df['nn_rmses'] * 100
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

    print(f"\nPlot of mean RMSE across all outputs saved as .pdf file to:\n   → {output_dir}.")

# Plot the results for the specific outputs
def Figure_6a_specific_outputs(options, df_specific, output_features_names):

    # Define plotting options for each of the outputs
    options_plot_mape={
        'y_axis':{
            'log_scale': False,
            'label': 'MAPE (\%)'
        },
        'x_axis':{
            'log_scale': True,
            'label': 'Number of Parameters',
            'x_min': 1e1,
            'x_max': 5e6,
            'x_ticks': [1e2, 1e4, 1e6],
        },
        'O2(X)': {
            'y_min': 0,
            'y_max': 350,
            'y_ticks': [1, 10, 100],
            'y_tick_labels': ['1', '10', '100'],
            'y_label_position': None,
            'y_over_axis_label': None,
        }
    }
    options_plot_mape['O2(+,X)'] = options_plot_mape['O2(X)']
    options_plot_mape['ne']      = options_plot_mape['O2(X)']
    
    # Define plotting options for each of the outputs
    options_plot_rmse={
        'y_axis':{
            'log_scale': False,
            'label': 'RMSE'
        },
        'x_axis':{
            'log_scale': True,
            'label': 'Number of Parameters',
            'x_min': 1e1,
            'x_max': 5e6,
            'x_ticks': [1e2, 1e4, 1e6],
        },
        'O2(X)': {
            'y_min': 0,
            'y_max': 0.5,
            'y_ticks': [1e-1, 2e-1, 3e-1, 4e-1],
            'y_tick_labels': ['1', '2', '3', '4'],
            'y_label_position': 0.52,
            'y_over_axis_label': r'($\times10^{-1}$)',
        },
        'O2(+,X)': {
            'y_min': 0,
            'y_max': 0.18,
            'y_ticks': [5e-2, 10e-2, 15e-2],
            'y_tick_labels': ['5', '10', '15'],
            'y_label_position': 0.188,
            'y_over_axis_label': r'($\times10^{-2}$)',
        },
        'ne': {
            'y_min': 0,
            'y_max': 0.18,
            'y_ticks': [5e-2, 10e-2, 15e-2],
            'y_tick_labels': ['5', '10', '15'],
            'y_label_position': 0.188,
            'y_over_axis_label': r'($\times10^{-2}$)',
        }
    }
    

    create_specific_output_plot(options, options_plot_mape, options_plot_rmse, df_specific, output_features_names, analysis = 'num_params')
    

#///////////////////////////////////////////////////////////////////////////////////////#
#///////////////////////////////// DATA SCALING STUDY //////////////////////////////////#
#///////////////////////////////////////////////////////////////////////////////////////#
# Main function to run the experiment
def run_data_scaling_study(config_original, large_dataset_path, dataset_sizes, options):
    print("\n")
    print("=" * 20 + "             Data Scaling Study             " + "=" * 20)
    
    return run_experiment(config_original, large_dataset_path, dataset_sizes, options)

# Plot the results for the mean of all outputs
def Figure_6e_mean_all_outputs(options, df):

    # Plot MAPE for NN and NN projection
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_xlabel('Dataset Size', fontsize=24)
    ax1.set_ylabel('MAPE (\%)', fontsize=24, fontweight='bold', labelpad=15)
    ax1.plot(df['dataset_sizes'], df['nn_mapes'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['dataset_sizes'], df['proj_mapes'], '-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection')
    ax1.tick_params(axis='y', labelsize=24, width = 2, length = 6)
    ax1.tick_params(axis='x', labelsize=24, width = 2, length = 6)
    #ax1.legend(loc='upper right', fontsize=20)
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAPE Variation Rate (\%)', fontsize=24, color='#B8B42D')
    improvement_rate = (df['proj_mapes'] - df['nn_mapes']) / df['nn_mapes'] * 100
    ax2.plot(df['dataset_sizes'], improvement_rate, '-o', color='#B8B42D', label='Improvement Rate (%)') 
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
    ax1.set_xlabel('Dataset Size', fontsize=24)
    ax1.set_ylabel('RMSE', fontsize=24, fontweight='bold', labelpad=15)
    #ax1.set_yscale('log')
    ax1.set_ylim(0, 25e-2)  # ax1.set_ylim(15e-3, 250e-3) 
    ax1.set_yticks([5e-2, 10e-2, 15e-2, 20e-2]) # ax1.set_yticks([25e-3, 50e-3, 100e-3, 200e-3])
    ax1.set_yticklabels(['5','10', '15', '20']) # ax1.set_yticklabels(['25','50', '100', '200'])
    # Plot lines without error bars
    ax1.plot(df['dataset_sizes'], df['nn_rmses'], '-o', color=models_parameters['NN']['color'], label='NN')
    ax1.plot(df['dataset_sizes'], df['proj_rmses'],'-o', color=models_parameters['proj_nn']['color'],linestyle='--', label='NN projection')
    # Add error bars to the plots
    #ax1.errorbar(df['dataset_sizes'], df['nn_rmses'], yerr=df['nn_rmse_uncertainties'],fmt='-o', color=models_parameters['NN']['color'], label='NN', capsize=5)
    #ax1.errorbar(df['dataset_sizes'], df['proj_rmses'], yerr=df['proj_rmse_uncertainties'],fmt='-o', color=models_parameters['proj_nn']['color'], linestyle='--', label='NN projection', capsize=5)
    ax1.tick_params(axis='y', labelsize=24, width = 2, length = 6)
    ax1.tick_params(axis='x', labelsize=24, width = 2, length = 6)
    #ax1.legend(loc='right', fontsize=20)
    # Add scientific notation label at the top of the y-axis
    plt.minorticks_off()
    ax1.text(0, 1.07, r'($\times10^{-3}$)', transform=plt.gca().transAxes, fontsize=24, ha='left', va='center')
    # Create a second y-axis for the improvement rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('RMSE Variation Rate (\%)', fontsize=24, color='#B8B42D')
    improvement_rate_rmse = (df['proj_rmses'] - df['nn_rmses']) / df['nn_rmses'] * 100
    ax2.plot(df['dataset_sizes'], improvement_rate_rmse, '-o', color='#B8B42D', label='Improvement Rate (%)')  # Set line color to gray
    ax2.tick_params(axis='y', labelsize=24, colors='#B8B42D', width = 2, length = 6)  # Set tick color to gray
    ax2.spines['right'].set_color('#B8B42D')  # Set color of the second y-axis spine to #B8B42D
    #fig.legend(loc='right', fontsize=18, ncol=3, bbox_to_anchor=(0.5, 1.1))
    ax1.spines['top'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['top'].set_linewidth(2)    # Save figure
    fig.tight_layout()
    output_dir = options['output_dir'] + "plots"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "mean_all_outputs_rmse.pdf")
    fig.savefig(save_path, pad_inches=0.2, format='pdf', dpi=300, bbox_inches='tight')

# Plot the results for the specific outputs
def Figure_6e_specific_outputs(options, df_specific, output_features_names):

    # get the min and max bounds for the y axis
    y_min_rmse, y_max_rmse = get_min_max_bounds(df_specific, 'rmses', log_scale = False)
    y_min_mape, y_max_mape = get_min_max_bounds(df_specific, 'mapes', log_scale = False)

    # Define plotting options for each of the outputs
    options_plot_mape={
        'y_axis':{
            'log_scale': False,
            'label': 'MAPE (\%)'
        },
        'x_axis':{
            'log_scale': False,
            'label': 'Dataset Size',
            'x_min': 1e1,
            'x_max': 5e6,
            'x_ticks': [1e2, 1e4, 1e6],
        },
        'O2(X)': {
            'y_min': y_min_mape,
            'y_max': y_max_mape,
            'y_ticks': [2, 8, 30],
            'y_tick_labels': ['2', '8', '30'],
            'y_label_position': None,
            'y_over_axis_label': None,
        }
    }
    options_plot_mape['O2(+,X)'] = options_plot_mape['O2(X)']
    options_plot_mape['ne']      = options_plot_mape['O2(X)']
    
    # Define plotting options for each of the outputs
    options_plot_rmse={
        'y_axis':{
            'log_scale': False,
            'label': 'RMSE'
        },
        'x_axis':{
            'log_scale': False,
            'label': 'Dataset Size',
            'x_min': 1e1,
            'x_max': 5e6,
            'x_ticks': [1e2, 1e4, 1e6],
        },
        'O2(X)': {
            'y_min': 0,
            'y_max': y_max_rmse,
            'y_ticks': [5e-2, 10e-2, 15e-2, 20e-2],
            'y_tick_labels': ['5', '10', '15', '20'],
            'y_label_position': y_max_rmse * 1.04,
            'y_over_axis_label': r'($\times10^{-2}$)',
        }
    }
    options_plot_rmse['O2(+,X)'] = options_plot_rmse['O2(X)']
    options_plot_rmse['ne']      = options_plot_rmse['O2(X)']
    
    create_specific_output_plot(options, options_plot_mape, options_plot_rmse, df_specific, output_features_names, analysis = 'dataset_sizes')

