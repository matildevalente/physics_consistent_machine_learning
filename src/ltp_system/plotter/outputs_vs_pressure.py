import os
import csv

import torch
import pickle

import numpy as np
import matplotlib as mpl
import torch.nn as nn
import random
from matplotlib.lines import Line2D

import pandas as pd
import matplotlib.pyplot as plt
from src.ltp_system.utils import set_seed, load_dataset, load_config
from src.ltp_system.utils import savefig, select_random_rows
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.ltp_system.data_prep import DataPreprocessor, LoadDataset, setup_dataset_with_preproprocessing_info
from src.ltp_system.pinn_nn import get_trained_bootstraped_models, load_checkpoints, NeuralNetwork, get_average_predictions, get_predictive_uncertainty
from src.ltp_system.projection import get_average_predictions_projected,constraint_p_i_ne

output_labels = [r'O$_2$(X)', r'O$_2$(a$^1\Delta_g$)', r'O$_2$(b$^1\Sigma_g^+$)', r'O$_2$(Hz)', r'O$_2^+$', r'O($^3P$)', r'O($^1$D)', r'O$^+$', r'O$^-$', r'O$_3$', r'O$_3^*$', r'$T_g$', r'T$_{nw}$', r'E$_{red}$', r'$v_d$', r'T$_{e}$', r'$n_e$']

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
}

plt.rcParams.update(pgf_with_latex)

# train the nn for a chosen architecture or load the parameters if it has been trained 
def get_trained_nn(config, data_preprocessing_info, idx_dataset, idx_sample, train_data, val_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'fig_6b_experiments', f'dataset_{idx_dataset}_sample_{idx_sample}')
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

# generate constant pressure inputs 
def generate_p_inputs(preped_data, normalized_inputs_):
    
    N_points = 1000
    p_max    = torch.max(torch.tensor(normalized_inputs_[:,0]))
    p_min    = torch.min(torch.tensor(normalized_inputs_[:,0]))
    i_fixed  = normalized_inputs_[:,1][1]
    R_fixed  = normalized_inputs_[:,2][1]

    step = (p_max - p_min) / (N_points - 1)
    input_data = np.array([[p_min + i * step, i_fixed , R_fixed ] for i in range(N_points)])
    
    return torch.tensor(input_data)

# compute the mape uncertainty of the aggregated nn models
def get_mape_uncertainty(normalized_targets, normalized_model_pred_uncertainty):
    # normalized_model_pred_uncertainty is the std / sqrt(n_models) of each model prediction
    # normalized_targets is the corresponding model target
    # here we compute the uncertaity of the mape by propagating the uncertainty of each model prediction

    # Compute MAPE uncertainty based on individual prediction uncertainties
    n = len(normalized_targets)
    mape_uncertainty = 0
    
    # Loop over each sample
    for i in range(n):
        # Extract the target row and uncertainty row for the current sample
        target_row = normalized_targets[i]
        uncertainty_row = normalized_model_pred_uncertainty[i]

        # Ensure that the target row has no zeros to avoid division errors
        non_zero_mask = target_row != 0  # Boolean mask where targets are non-zero

        # Use the mask to compute the uncertainty for valid (non-zero target) values
        mape_uncertainty += torch.sum((uncertainty_row[non_zero_mask] / target_row[non_zero_mask]) ** 2).item()

    # Final uncertainty
    mape_uncertainty = np.sqrt(mape_uncertainty) / n

    return mape_uncertainty

# the the mape and mape uncertainty of the nn aggregated model
def evaluate_model(normalized_model_predictions, normalized_targets, normalized_model_pred_uncertainty):
    
    # compute the mape and sem with respect to target
    mape = mean_absolute_percentage_error(normalized_targets, normalized_model_predictions)
    mape_uncertainty = get_mape_uncertainty(normalized_targets, normalized_model_pred_uncertainty)
    rmse = np.sqrt(mean_squared_error(normalized_targets, normalized_model_predictions))

    return mape, mape_uncertainty, normalized_model_predictions, rmse

# get the mape of the nn projected predictions
def evaluate_projection(normalized_proj_predictions, normalized_targets):

    # compute the mape and sem with respect to target
    mape = mean_absolute_percentage_error(normalized_targets, normalized_proj_predictions)
    rmse = np.sqrt(mean_squared_error(normalized_targets, normalized_proj_predictions))

    return mape, rmse

# get the data for the Figure 6b
def get_data_Figure_6b(networks, file_path, w_matrix, data_preprocessing_info):
  
    # /// 1. EXTRACT DATASET & PREPROCESS THE DATASET///
    normalized_inputs, normalized_targets = load_data(file_path, data_preprocessing_info)

    # get the normalized model predictions
    normalized_inputs_ = normalized_inputs.clone() 
    normalized_model_predictions_simul =  get_average_predictions(networks, normalized_inputs_)
    normalized_model_pred_uncertainty_simul =  get_predictive_uncertainty(networks, normalized_inputs_) 
    normalized_proj_predictions_simul  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions_simul), normalized_inputs_, data_preprocessing_info, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions_simul = torch.tensor(np.stack(normalized_proj_predictions_simul))

    # generate constant pressure inputs 
    normalized_inputs_contiuous_p = generate_p_inputs(data_preprocessing_info, normalized_inputs_)
    
    # compute 
    normalized_model_predictions_contiuous_p =  get_average_predictions(networks, normalized_inputs_contiuous_p)
    normalized_model_pred_uncertainty_contiuous_p =  get_predictive_uncertainty(networks, normalized_inputs_contiuous_p)
    normalized_proj_predictions_contiuous_p = get_average_predictions_projected(normalized_model_predictions_contiuous_p, normalized_inputs_contiuous_p, data_preprocessing_info, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions_contiuous_p = torch.tensor(np.stack(normalized_proj_predictions_contiuous_p))

    # Inverse Normalization and Log Transform
    denormalized_inputs_simul, denormalized_targets_simul = data_preprocessing_info.inverse_transform(normalized_inputs_, normalized_targets)
    _, denormalized_proj_predictions_simul          = data_preprocessing_info.inverse_transform(normalized_inputs_, normalized_proj_predictions_simul)
    _, denormalized_model_predictions_simul         = data_preprocessing_info.inverse_transform(normalized_inputs_, normalized_model_predictions_simul)
    #
    denormalized_inputs_contiuous_p, denormalized_model_predictions_contiuous_p = data_preprocessing_info.inverse_transform(normalized_inputs_contiuous_p, normalized_model_predictions_contiuous_p)
    denormalized_inputs_contiuous_p, denormalized_proj_predictions_contiuous_p = data_preprocessing_info.inverse_transform(normalized_inputs_contiuous_p, normalized_proj_predictions_contiuous_p)

    #
    mape_nn, mape_uncertainty_nn, normalized_model_predictions_nn, rmse_nn = evaluate_model(normalized_targets, normalized_model_predictions_simul, normalized_model_pred_uncertainty_simul)
    mape_proj, rmse_proj = evaluate_projection(normalized_proj_predictions_simul, normalized_targets)
    
    denormalized_predictions_dict = {
       'discrete_inputs'             : denormalized_inputs_simul, 
       'discrete_targets'            : denormalized_targets_simul,
       'discrete_nn_predictions'     : denormalized_model_predictions_simul,
       'discrete_nn_proj_predictions': denormalized_proj_predictions_simul,

       'const_p_inputs'              : denormalized_inputs_contiuous_p, 
       'nn_model_outputs'            : denormalized_model_predictions_contiuous_p, 
       'nn_model_pred_uncertainties' : normalized_model_pred_uncertainty_contiuous_p, 
       'nn_proj_outputs'             : denormalized_proj_predictions_contiuous_p, 
    }
    errors_dict = {
       'mape_nn':mape_nn, 
       'rmse_nn': rmse_nn, 
       'mape_proj':mape_proj,
       'rmse_proj':rmse_proj
    }

    return denormalized_predictions_dict, errors_dict

# append the error values to the error data
def append_error_values(error_data, output, discrete_targets, discrete_nn_predictions, discrete_nn_proj_predictions):
    # Convert targets and predictions to numpy array and make copies
    discrete_targets_ = np.array(discrete_targets)
    discrete_nn_predictions_ = np.array(discrete_nn_predictions)
    discrete_nn_proj_predictions_ = np.array(discrete_nn_proj_predictions)

    # Compute MAPE for each prediction
    mape_nn = np.mean(np.abs((discrete_targets_ - discrete_nn_predictions_) / discrete_targets_)) * 100
    mape_nn_proj = np.mean(np.abs((discrete_targets_ - discrete_nn_proj_predictions_) / discrete_targets_)) * 100
    
    # Compute RMSE for each prediction
    rmse_nn = np.sqrt(np.mean((discrete_targets_ - discrete_nn_predictions_) ** 2))
    rmse_nn_proj = np.sqrt(np.mean((discrete_targets_ - discrete_nn_proj_predictions_) ** 2))

    # Append error values as a dictionary
    error_data.append({
        "Output": output,
        "MAPE NN": mape_nn,
        "MAPE NN Projection": mape_nn_proj,
        "RMSE NN": rmse_nn,
        "RMSE NN Projection": rmse_nn_proj
    })

# physical plot - outputs vs. pressure at a given discharge current (1 plot)
#def Figure_6b(config, denormalized_predictions_dict, test_case):
def Figure_6b(config, df_discrete, df_continuum, test_case):
    # extract output features
    outputs = config['dataset_generation']['output_features']
    n_inputs = 3
    n_outputs = 17

    # inputs and targets based on the given file (simulation points)
    discrete_inputs  =  df_discrete[[f'input_{i+1}' for i in range(n_inputs)]].to_numpy()
    discrete_targets =  df_discrete[[f'target_{i+1}' for i in range(n_outputs)]].to_numpy()
    discrete_nn_predictions = df_discrete[[f'nn_pred_{i+1}' for i in range(n_outputs)]].to_numpy()
    discrete_nn_proj_predictions = df_discrete[[f'nn_proj_pred_{i+1}' for i in range(n_outputs)]].to_numpy()

    # inputs and targets continuous distribution
    const_p_inputs   =  df_continuum[[f'input_{i+1}' for i in range(n_inputs)]].to_numpy()
    nn_model_outputs =  df_continuum[[f'nn_pred_{i+1}' for i in range(n_outputs)]].to_numpy()
    nn_model_pred_uncertainties  =  df_continuum[[f'nn_pred_uncertainty_{i+1}' for i in range(n_outputs)]].to_numpy()
    nn_proj_outputs = df_continuum[[f'nn_proj_pred_{i+1}' for i in range(n_outputs)]].to_numpy()

    error_normalized_data = []
    error_denormalized_data = []
    for i, output in enumerate(outputs):
        
        # compare the loki target points to the predictions by the NN
        append_error_values(error_normalized_data, output, discrete_targets[:,i], discrete_nn_predictions[:,i], discrete_nn_proj_predictions[:,i])
        append_error_values(error_denormalized_data, output, discrete_targets[:,i], discrete_nn_predictions[:,i], discrete_nn_proj_predictions[:,i])

        # Resetting the current figure
        plt.clf()  # Clear the current figure
        mpl.use('pgf')
        plt.style.use('seaborn-v0_8-paper')

        # Plot 1: Scatter plot comparing ne_model and calculated ne
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.legend(fontsize='large', loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=15, width=2)
        ax.tick_params(axis='both', which='minor', labelsize=15, width=2)
        ax.xaxis.set_tick_params(which='both', direction='in', top=True, bottom=True)
        ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
        ax.yaxis.get_offset_text().set_fontsize(24)

        zero_errors = np.zeros_like(nn_model_pred_uncertainties[:,i])
        ax.errorbar(const_p_inputs[:,0], nn_model_outputs[:,i], yerr=None, xerr=nn_model_pred_uncertainties[:,i], fmt='ro', color='blue', ecolor='lightblue', capsize=4, markersize=3,capthick=2)
        ax.plot([], [], '-', color='blue', markersize=5, label=f'NN', linewidth=3)
        ax.errorbar(const_p_inputs[:,0], nn_proj_outputs[:,i], yerr=zero_errors, xerr=zero_errors, fmt='ro', color='green', ecolor='lightgreen', capsize=0, markersize=3,capthick=2)
        ax.plot([], [], '--', color='green', markersize=5, label=f'NN projection', linewidth=3)
        ax.plot(discrete_inputs[:,0], discrete_targets[:,i], 'x', markeredgecolor='red',markeredgewidth=2, color='red', label=f'LoKI (target)', markersize=10, zorder=10)

        # Combine all custom handles for the legend
        ax.legend(fontsize="24")

        ax.set_xlabel(r"Pressure (Pa)", fontsize=24, fontweight='bold')
        ax.set_ylabel(output_labels[i], fontsize=24, fontweight='bold')

        # Make axis numbers bold
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(24)
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)

        plt.tight_layout()

        # Save figures
        output_dir = config['plotting']['output_dir'] + "/Figures_6b/" + "test_case_" + str(test_case) + "/"
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{output}.pdf")
        plt.savefig(save_path, pad_inches = 0.2)
        plt.close()

    # Convert error data to a DataFrame and save to local directory
    error_denormalized_df = pd.DataFrame(error_denormalized_data)
    csv_save_path = os.path.join(output_dir, "error_denormalized_data.csv")
    error_denormalized_df.to_csv(csv_save_path, index=False)
    
    # Convert error data to a DataFrame and save to local directory
    error_normalized_df = pd.DataFrame(error_normalized_data)
    csv_save_path = os.path.join(output_dir, "error_normalized_data.csv")
    error_normalized_df.to_csv(csv_save_path, index=False)

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

"""# run the experiments for the Figure 6b
def run_fig_6b_experiments(config, options, dataset_sizes, target_data_file_path, large_dataset_path, testcase, seed_):

    # create the preprocessed_data object - all the smaller datasets should use the same scalers
    _, large_dataset = load_dataset(config, large_dataset_path)
    data_preprocessing_info = DataPreprocessor(config)
    data_preprocessing_info.setup_dataset(large_dataset.x, large_dataset.y)  
    
    for idx_dataset, dataset_size in enumerate(dataset_sizes):
        # 1. generate config using the configuration from the options
        config_experiment = generate_config_(config, options)

        # 2. create a dataset with the given number of rows
        temp_file = select_random_rows(large_dataset_path, dataset_size, seed=seed_)

        # 3. read the dataset and preprocess data
        _, temp_dataset = load_dataset(config_experiment, temp_file)
        train_data_norm, _, val_data_norm  = setup_dataset_with_preproprocessing_info(temp_dataset.x, temp_dataset.y, data_preprocessing_info)  

        # 4. create the val loader
        val_loader = torch.utils.data.DataLoader(val_data_norm, batch_size=config_experiment['nn_model']['batch_size'], shuffle=True)

        # 5. train the neural network model (nn)
        nn_models, _, _, _ = get_trained_nn(config_experiment, data_preprocessing_info, testcase, 0 ,train_data_norm, val_loader)      

        # 6. get the results 
        preds_dict, errors_dict = get_data_Figure_6b(config_experiment, nn_models, target_data_file_path, torch.eye(17), data_preprocessing_info)

        return preds_dict, errors_dict, config_experiment

"""

# Figures_6a; output/ltp_system/checkpoints/different_datasets/dataset_{}_sample_{}
def run_figures_output_vs_pressure_diff_datasets(config, options, dataset_sizes_list, target_data_file_path):

    """
    For each dataset size, different models were trained with different samples. 
    This function performs the predictions of constant I and R for varying P using this pre-trained models with different dataset sizes. 
    The final (R,I)-constant curve prediction is a mean of the predictions of the different models trained with different samples of the same size.
    Different samples are used to reduce the bias of the datapoints selection.
    """

    # extract the number of samples per dataset size
    n_samples = options['n_samples']
    n_inputs = 3  # number of input features
    n_outputs = 17  # number of target features
    
    # loop over the different dataset lenghts
    for idx_dataset, dataset_size in enumerate(range(len(dataset_sizes_list))):

        # Initialize empty DataFrame with the column names
        columns_discrete = [
            *[f'input_{i+1}' for i in range(n_inputs)],
            *[f'target_{i+1}' for i in range(n_outputs)],
            *[f'nn_pred_{i+1}' for i in range(n_outputs)],
            *[f'nn_proj_pred_{i+1}' for i in range(n_outputs)]
        ]
        columns_continuum = [
            *[f'input_{i+1}' for i in range(n_inputs)],
            *[f'nn_pred_{i+1}' for i in range(n_outputs)],
            *[f'nn_pred_uncertainty_{i+1}' for i in range(n_outputs)],
            *[f'nn_proj_pred_{i+1}' for i in range(n_outputs)]
        ]
        df_discrete  = pd.DataFrame(columns=columns_discrete)
        df_continuum = pd.DataFrame(columns=columns_continuum)

        # loop over the different samples
        for idx_sample in range(n_samples):

            # directory where the trained models are stored
            checkpoint_dir = f"output/ltp_system/checkpoints/different_datasets/dataset_{idx_dataset}_sample_{idx_sample}"
            
            # load the already trained model from the checkpoint dir
            try:
                options['learning_rate'] = config['nn_model']['learning_rate']
                options['batch_size'] = config['nn_model']['batch_size']
                options['lambda_physics'] = config['nn_model']['lambda_physics']
                nn_models, _, hidden_sizes, activation_fns, training_time = load_checkpoints(options, NeuralNetwork, checkpoint_dir)
                
                # Load the preprocessing_info object with the information about the scalers
                file_path = "output/ltp_system/checkpoints/different_datasets/data_preprocessing_info.pkl"
                with open(file_path, 'rb') as file:
                    data_preprocessing_info = pickle.load(file)                
                    
            except FileNotFoundError:
                raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
            
            # Get the constant-current predictions for the specific model get_data_Figure_6b(networks, file_path, w_matrix, data_preprocessing_info)
            predictions_dict_sample_idx, errors_dict_sample_idx = get_data_Figure_6b(nn_models, target_data_file_path, torch.eye(17), data_preprocessing_info)
            
            # inputs and targets based on the given file (simulation points)
            discrete_inputs  =  np.array(predictions_dict_sample_idx['discrete_inputs'])
            discrete_targets =  np.array(predictions_dict_sample_idx['discrete_targets'])
            discrete_nn_predictions = np.array(predictions_dict_sample_idx['discrete_nn_predictions'])
            discrete_nn_proj_predictions = np.array(predictions_dict_sample_idx['discrete_nn_proj_predictions'])

            # inputs and targets continuous distribution
            p_inputs   =  np.array(predictions_dict_sample_idx['const_p_inputs'])
            nn_pred =  np.array(predictions_dict_sample_idx['nn_model_outputs'])
            nn_proj_pred  =  np.array(predictions_dict_sample_idx['nn_proj_outputs'])
            nn_pred_uncertainty = np.array(predictions_dict_sample_idx['nn_model_pred_uncertainties'])

            # append the discrete results for this sample
            n_discrete_points = len(discrete_inputs[:,0])
            new_data = pd.DataFrame(
                np.hstack([
                    discrete_inputs,
                    discrete_targets,
                    discrete_nn_predictions,
                    discrete_nn_proj_predictions
                ]),
                columns=columns_discrete,
                index=range(n_discrete_points) 
            )
            df_discrete = pd.concat([df_discrete, new_data])
            
            # append the continuum results for this sample
            n_continuum_points = len(p_inputs[:,0])
            new_data = pd.DataFrame(
                np.hstack([
                    p_inputs,
                    nn_pred,
                    nn_pred_uncertainty,
                    nn_proj_pred
                ]),
                columns=columns_continuum,
                index=range(n_continuum_points) 
            )
            df_continuum = pd.concat([df_continuum, new_data])
            
        # Group by the cycling index (0-n_discrete_points) and calculate mean
        df_aggregated_discrete_predictions_dataset_idx = df_discrete.groupby(df_discrete.index % n_discrete_points).mean()

        # Group by the cycling index (0-49) and calculate mean
        df_aggregated_continuum_predictions_dataset_idx = df_continuum.groupby(df_continuum.index % n_continuum_points).mean()

        # make the plot
        Figure_6b(config, df_aggregated_discrete_predictions_dataset_idx, df_aggregated_continuum_predictions_dataset_idx, test_case = f"different_datasets/dataset_{dataset_size}")


# 
def run_figures_output_vs_pressure_diff_architectures(config, options, target_data_file_path, architectures_file_path):

    """
    For each dataset size, different models were trained with different samples. 
    This function performs the predictions of constant I and R for varying P using this pre-trained models with different dataset sizes. 
    The final (R,I)-constant curve prediction is a mean of the predictions of the different models trained with different samples of the same size.
    Different samples are used to reduce the bias of the datapoints selection.
    """

    checkpoints_folder = options['checkpoints_dir']

    # extract the number of samples per dataset size
    n_inputs = 3  # number of input features
    n_outputs = 17  # number of target features

    # load the nn architectures from a file
    random_architectures_list = []
    with open(architectures_file_path, mode='r') as file:
        reader = csv.reader(file)
        random_architectures_list = [row for row in reader]
    random_architectures_list = [[int(num) for num in sublist] for sublist in random_architectures_list]

    # loop over the different dataset lenghts
    for idx_architecture, architecture in enumerate(random_architectures_list):

        # Initialize empty DataFrame with the column names
        columns_discrete = [
            *[f'input_{i+1}' for i in range(n_inputs)],
            *[f'target_{i+1}' for i in range(n_outputs)],
            *[f'nn_pred_{i+1}' for i in range(n_outputs)],
            *[f'nn_proj_pred_{i+1}' for i in range(n_outputs)]
        ]
        columns_continuum = [
            *[f'input_{i+1}' for i in range(n_inputs)],
            *[f'nn_pred_{i+1}' for i in range(n_outputs)],
            *[f'nn_pred_uncertainty_{i+1}' for i in range(n_outputs)],
            *[f'nn_proj_pred_{i+1}' for i in range(n_outputs)]
        ]
        df_discrete  = pd.DataFrame(columns=columns_discrete)
        df_continuum = pd.DataFrame(columns=columns_continuum)

        # directory where the trained models are stored
        checkpoint_dir = f"{checkpoints_folder}architecture_{idx_architecture}"
        
        # load the already trained model from the checkpoint dir
        try:
            options['hidden_sizes'] = architecture
            options['activation_fns'] = options['activation_func']
            options['learning_rate'] = config['nn_model']['learning_rate']
            options['batch_size'] = config['nn_model']['batch_size']
            options['lambda_physics'] = config['nn_model']['lambda_physics']
            nn_models, _, _, _, _ = load_checkpoints(options, NeuralNetwork, checkpoint_dir)
            
            # Load the preprocessing_info object with the information about the scalers
            file_path = f"{checkpoints_folder}data_preprocessing_info.pkl"
            with open(file_path, 'rb') as file:
                data_preprocessing_info = pickle.load(file)                
                
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
        # Get the constant-current predictions for the specific model get_data_Figure_6b(networks, file_path, w_matrix, data_preprocessing_info)
        predictions_dict_sample_idx, errors_dict_sample_idx = get_data_Figure_6b(nn_models, target_data_file_path, torch.eye(17), data_preprocessing_info)
        
        # inputs and targets based on the given file (simulation points)
        discrete_inputs  =  np.array(predictions_dict_sample_idx['discrete_inputs'])
        discrete_targets =  np.array(predictions_dict_sample_idx['discrete_targets'])
        discrete_nn_predictions = np.array(predictions_dict_sample_idx['discrete_nn_predictions'])
        discrete_nn_proj_predictions = np.array(predictions_dict_sample_idx['discrete_nn_proj_predictions'])

        # inputs and targets continuous distribution
        p_inputs   =  np.array(predictions_dict_sample_idx['const_p_inputs'])
        nn_pred =  np.array(predictions_dict_sample_idx['nn_model_outputs'])
        nn_proj_pred  =  np.array(predictions_dict_sample_idx['nn_proj_outputs'])
        nn_pred_uncertainty = np.array(predictions_dict_sample_idx['nn_model_pred_uncertainties'])

        # append the discrete results for this sample
        n_discrete_points = len(discrete_inputs[:,0])
        new_data = pd.DataFrame(
            np.hstack([
                discrete_inputs,
                discrete_targets,
                discrete_nn_predictions,
                discrete_nn_proj_predictions
            ]),
            columns=columns_discrete,
            index=range(n_discrete_points) 
        )
        df_discrete = pd.concat([df_discrete, new_data])
        
        # append the continuum results for this sample
        n_continuum_points = len(p_inputs[:,0])
        new_data = pd.DataFrame(
            np.hstack([
                p_inputs,
                nn_pred,
                nn_pred_uncertainty,
                nn_proj_pred
            ]),
            columns=columns_continuum,
            index=range(n_continuum_points) 
        )
        df_continuum = pd.concat([df_continuum, new_data])
            
        # make the plot
        n_hidden_layers = options['n_hidden_layers']
        Figure_6b(config, df_discrete, df_continuum, test_case = f"different_architectures/{n_hidden_layers}_hidden_layers/architecture_{idx_architecture}")