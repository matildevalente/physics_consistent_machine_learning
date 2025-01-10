import os
import re
import yaml
import glob
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, Any, Tuple

from src.ltp_system.utils import set_seed, load_dataset, load_config, select_random_rows, split_dataset
from src.ltp_system.data_prep import DataPreprocessor, setup_dataset_with_preproprocessing_info
from src.ltp_system.plotter.eda import apply_eda
from src.ltp_system.pinn_nn import get_trained_bootstraped_models, load_checkpoints, NeuralNetwork
from src.ltp_system.projection import get_inverse_covariance_matrix , print_matrix, compute_projection_results, compute_mape_physical_laws, compute_errors_physical_laws_loki, compute_rmse_physical_laws
from src.ltp_system.plotter.loss_curves import loss_curves
from src.ltp_system.plotter.barplots import Figure_4d, Figure_4a, Figure_4b
from src.ltp_system.plotter.Figure_6a import run_experiment_6a, Figure_6a_mean_all_outputs, Figure_6a_specific_outputs
from src.ltp_system.plotter.Figure_6e import Figure_6e, run_experiment_6e
from src.ltp_system.plotter.outputs_vs_pressure import get_data_Figure_6b, Figure_6b, run_fig_6b_experiments

output_labels = [r'O$_2$(X)', r'O$_2$(a$^1\Delta_g$)', r'O$_2$(b$^1\Sigma_g^+$)', r'O$_2$(Hz)', r'O$_2^+$', r'O($^3P$)', r'O($^1$D)', r'O$^+$', r'O$^-$', r'O$_3$', r'O$_3^*$', r'$T_g$', r'T$_{nw}$', r'$E/N$', r'$v_d$', r'T$_{e}$', r'$n_e$']

def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # Set seed 
        set_seed(42)        

        # Load the configuration file
        config = load_config('configs/ltp_system_config.yaml')

        # /// 1. EXTRACT FULL DATASET ///
        large_dataset_path = 'data/ltp_system/data_3000_points.txt'
        _, large_dataset = load_dataset(config, large_dataset_path)
        
        # /// 2. PREPROCESS THE FULL DATASET ///
        data_preprocessing_info = DataPreprocessor(config)
        data_preprocessing_info.setup_dataset(large_dataset.x, large_dataset.y) 
        testing_file, training_file = split_dataset(large_dataset_path, n_testing_points = 300)     
        #apply_eda(config, data_preprocessing_info, large_dataset.y)
        
        # /// 3. TRAIN THE NEURAL NETWORK (NN) ///
        nn_models, nn_losses_dict, device_nn = get_trained_nn(config, data_preprocessing_info, training_file, dataset_size = 1000)
        loss_curves(config['nn_model'], config['plotting'], nn_losses_dict)
    
        # /// 4. TRAIN THE PHYSICS-INFORMED NEURAL NETWORK (PINN) ///
        pinn_models, pinn_losses_dict, device_pinn = get_trained_pinn(config, data_preprocessing_info, training_file, dataset_size = 1000)
        loss_curves(config['pinn_model'], config['plotting'], pinn_losses_dict)

        # /// 5. TESTSET RESULTS OF NN, PINN and PROJECTION APPLIED TO BOTH MODELS PREDICTIONS /// 
        saving_dir = 'src/ltp_system/figures/Figures_4/Figure_4b/'
        for error_type in ['mape', 'rmse']:
            # Performances on output predictions
            nn_error_sem_dict = compute_projection_results(config['nn_model'], torch.eye(17), testing_file, data_preprocessing_info, nn_models, error_type)
            pinn_error_sem_dict = compute_projection_results(config['pinn_model'], torch.eye(17), testing_file, data_preprocessing_info, pinn_models, error_type)
            Figure_4a(config['plotting'], nn_error_sem_dict, pinn_error_sem_dict, error_type)
            Figure_4d(nn_error_sem_dict, config['nn_model'], config['plotting'], error_type)  

            # Performances on compliance with physical laws
            laws_dict = get_laws_dict(testing_file, data_preprocessing_info, nn_models, pinn_models, saving_dir, error_type)
            Figure_4b(config['plotting'], laws_dict, error_type)

        # /// 7. NN and NN_Proj errors as a func of the model's parameters + NN training time /// 
        options = {
            'n_bootstrap_models': 30,
            'activation_func': 'leaky_relu', 
            'APPLY_EARLY_STOPPING': True,
            'num_epochs': 100000,            # Along w/ the early stopping, we use a high max epochs to ensure convergence 
            'max_hidden_layers': 1, 
            'min_hidden_layers':1,
            'max_neurons_per_layer': 500, 
            'min_neurons_per_layer': 1, 
            'n_steps': 35,
            'w_matrix': torch.eye(17), 
            'PRINT_LOSS_VALUES': True,
            'RETRAIN_MODEL': False,
            'extract_results_specific_outputs': ['O2(X)', 'O2(+,X)', 'ne'],
            'patience': 2,
            'alpha': 0.0001,
            'output_dir': 'src/ltp_system/figures/Figures_6a/'         # dir for tables and plots
        }
        # select 'data_size' random rows from the training file to train the different nn architectures
        temp_file = select_random_rows(training_file, n = 600)
        # run the experiment and obtain results as pandas dataframes
        df_results_6a_all, df_results_6a_specific = run_experiment_6a(config, temp_file, options)
        # plot the results
        Figure_6a_mean_all_outputs(options, df_results_6a_all)
        Figure_6a_specific_outputs(options, df_results_6a_all, df_results_6a_specific, output_labels)

        """# /// 8. NN and NN_Proj errors as a func of the dataset size + approximate loki computation time /// 
        options = {
            'n_samples': 30, 
            'APPLY_EARLY_STOPPING': True,
            'num_epochs':  100000000,
            'n_bootstrap_models': 5, 
            'hidden_sizes': [451, 315, 498, 262],
            'activation_fns': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'], 
            'w_matrix': torch.eye(17), 
            'PRINT_LOSS_VALUES': False,
            'RETRAIN_MODEL': False, 
        }
        # Get the list of files to analyze
        dataset_sizes = [175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
        #df_results_6e = run_experiment_6e(config, training_file, dataset_sizes, options)
        #print(df_results_6e)
        #Figure_6e(config['plotting'], df_results_6e)
        
        # /// 9. CONST CURRENT PLOTS ///
        dataset_sizes = [1000]
        options = {
            'num_epochs':  100,
            'APPLY_EARLY_STOPPING': False,
            'n_bootstrap_models': 30, 
            'hidden_sizes': [451, 315, 498, 262],
            'activation_fns': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'],
            'w_matrix': torch.eye(17), 
            'RETRAIN_MODEL': True, 
            'PRINT_LOSS_VALUES': False
        }
        training_file = 'data/ltp_system/data_1000_points.txt'
        target_data_file_path = 'data/ltp_system/const_current/data_50_points_30mA.txt'
        #run_fig_6b_experiments(config, options, dataset_sizes, target_data_file_path, training_file, testcase = "a")
        
        options['num_epochs'] = 50
        options['n_bootstrap_models'] = 1
        #run_fig_6b_experiments(config, options, dataset_sizes, target_data_file_path, training_file, testcase = "b")

        options['num_epochs'] = 5
        options['n_bootstrap_models'] = 1
        #run_fig_6b_experiments(config, options, dataset_sizes, target_data_file_path, training_file, testcase = "c")

        dataset_sizes = [300]
        options['num_epochs'] = 5
        options['n_bootstrap_models'] = 1
        training_file = 'data/ltp_system/data_1000_points.txt'
        #run_fig_6b_experiments(config, options, dataset_sizes, target_data_file_path, training_file, testcase = "d")"""

        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


def get_trained_nn(config, data_preprocessing_info, training_file, dataset_size):
    # create checkpoints directory for the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'nn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # read from the train_path file and randomly select 'dataset_size' rows
    temp_file = select_random_rows(training_file, dataset_size)

    # read the dataset and preprocess data
    _, temp_dataset = load_dataset(config, temp_file)
    train_data, _, val_data  = setup_dataset_with_preproprocessing_info(temp_dataset.x, temp_dataset.y, data_preprocessing_info)  

    # create the val loader
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['nn_model']['batch_size'], shuffle=True)

    if config['nn_model']['RETRAIN_MODEL']:

        nn_models, nn_losses_dict, _ = get_trained_bootstraped_models(config['nn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
        return nn_models, nn_losses_dict, device
    else:
        try:
            nn_models, nn_losses_dict, _, _, _ = load_checkpoints(config['nn_model'], NeuralNetwork, checkpoint_dir)

            # Check if any losses were loaded
            if (nn_losses_dict['losses_train_mean'].size == 0 or 
                nn_losses_dict['losses_val_mean'].size == 0 or 
                nn_losses_dict['epoch'].size == 0):
                print("Warning: Incomplete loss history found in checkpoint. Retraining the model.")
                nn_models, nn_losses_dict, _ = get_trained_bootstraped_models(config['nn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
            else:
                print("Checkpoint loaded successfully with loss history.")

        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")

    return nn_models, nn_losses_dict, device


def get_trained_pinn(config, data_preprocessing_info, training_file, dataset_size):
    # create checkpoints directory for the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'pinn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # read from the train_path file and randomly select 'dataset_size' rows
    temp_file = select_random_rows(training_file, dataset_size)

    # read the dataset and preprocess data
    _, temp_dataset = load_dataset(config, temp_file)
    train_data, _, val_data  = setup_dataset_with_preproprocessing_info(temp_dataset.x, temp_dataset.y, data_preprocessing_info)  

    # create the val loader
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['nn_model']['batch_size'], shuffle=True)

    if config['pinn_model']['RETRAIN_MODEL']:

        pinn_models, pinn_losses_dict, _ = get_trained_bootstraped_models(config['pinn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
        return pinn_models, pinn_losses_dict, device
    else:
        try:
            pinn_models, pinn_losses_dict, _, _, _ = load_checkpoints(config['pinn_model'], NeuralNetwork, checkpoint_dir)
        
            # Check if any losses were loaded
            if (pinn_losses_dict['losses_train_mean'].size == 0 or 
                pinn_losses_dict['losses_val_mean'].size == 0 or 
                pinn_losses_dict['epoch'].size == 0):
                print("Warning: Incomplete loss history found in checkpoint. Retraining the model.")
                pinn_models, pinn_losses_dict, _ = get_trained_bootstraped_models(config['pinn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
            else:
                print("Checkpoint loaded successfully with loss history.")     
        
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
    return pinn_models, pinn_losses_dict, device


def get_laws_dict(file_name, preprocessed_data, nn_models, pinn_models, saving_dir, error_type):
    if error_type == 'mape':
        nn_laws_dict   = compute_mape_physical_laws(file_name, preprocessed_data, nn_models, torch.eye(17), "nn_model")
        pinn_laws_dict = compute_mape_physical_laws(file_name, preprocessed_data, pinn_models, torch.eye(17), "pinn_model")
    elif error_type == 'rmse':
        nn_laws_dict   = compute_rmse_physical_laws(file_name, preprocessed_data, nn_models, torch.eye(17), "nn_model")
        pinn_laws_dict = compute_rmse_physical_laws(file_name, preprocessed_data, pinn_models, torch.eye(17), "pinn_model")
    else:
        raise ValueError(f"Invalid error type: {error_type}. Please choose 'mape' or 'rmse'.") 

    # compute the errors in compliance with physical laws for the loki model
    loki_laws_dict = compute_errors_physical_laws_loki(file_name, preprocessed_data, torch.eye(17), error_type)

    # merge the dictionaries
    laws_dict = {**nn_laws_dict, **pinn_laws_dict, **loki_laws_dict}

    # get the keys of the first dictionary
    keys = list(nn_laws_dict['nn_model'].keys())
    
    # Save results to local directory
    os.makedirs(saving_dir, exist_ok=True)
    file_path = os.path.join(saving_dir, f'physical_compliance_{error_type}.csv')
    pd.options.display.float_format = '{:.3e}'.format
    df = pd.DataFrame(laws_dict, index=keys)
    df.to_csv(file_path, index=True, float_format='%.3e')

    return laws_dict

def get_inputs_from_loader(loader):
    x_test = np.concatenate([X.cpu().numpy() for X, *_ in loader])
    #print(x_test)

    return x_test



if __name__ == "__main__":
    main()










