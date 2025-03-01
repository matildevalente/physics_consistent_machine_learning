import os
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict, Any, Tuple

from clean import flush_model_artifacts
from src.ltp_system.plotter.eda import apply_eda
from src.ltp_system.utils import set_seed, load_dataset, load_config, select_random_rows, sample_dataset
from src.ltp_system.data_prep import DataPreprocessor, setup_dataset_with_preproprocessing_info
from src.ltp_system.pinn_nn import get_trained_bootstraped_models, load_checkpoints, NeuralNetwork
from src.ltp_system.projection import compute_projection_results, compute_mape_physical_laws, compute_errors_physical_laws_loki, compute_rmse_physical_laws
from src.ltp_system.plotter.loss_curves import loss_curves
from src.ltp_system.plotter.barplots import Figure_4d, Figure_4a, Figure_4b
from src.ltp_system.plotter.scaling_studies import run_data_scaling_study, run_ablation_study_architectures, Figure_6a_mean_all_outputs, Figure_6a_specific_outputs, Figure_6e_mean_all_outputs, Figure_6e_specific_outputs, Figure_computation_times
from src.ltp_system.plotter.outputs_vs_pressure import run_figures_output_vs_pressure_diff_datasets, run_figures_output_vs_pressure_diff_architectures


output_labels = [r'O$_2$(X)', r'O$_2$(a$^1\Delta_g$)', r'O$_2$(b$^1\Sigma_g^+$)', r'O$_2$(Hz)', r'O$_2^+$', r'O($^3P$)', r'O($^1$D)', r'O$^+$', r'O$^-$', r'O$_3$', r'O$_3^*$', r'$T_g$', r'T$_{nw}$', r'$E/N$', r'$v_d$', r'T$_{e}$', r'$n_e$']


def main(retrain_flag):
    try:
        # /// 1. SETUP LOGGING ///
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # /// 2. SET SEED ///
        set_seed(42)        

        # /// 3. LOAD CONFIGURATION FILE /// 
        config = load_config(retrain_flag, 'configs/ltp_system_config.yaml')

        # /// 4. EXTRACT FULL DATASET ///
        large_dataset_path = 'data/ltp_system/data_3000_points.txt'
        _, large_dataset = load_dataset(config, large_dataset_path)
        
        # /// 5. PREPROCESS THE FULL DATASET ///
        dataset_size = 1000
        data_preprocessing_info = DataPreprocessor(config)
        data_preprocessing_info.setup_dataset(large_dataset.x, large_dataset.y) 
        testing_file, training_file = sample_dataset(large_dataset_path, n_testing_points = 300)  
        apply_eda(config, data_preprocessing_info, large_dataset.y)
        train_data, val_loader = preprocess_and_split(config, training_file, dataset_size, data_preprocessing_info)

        # /// 6. TRAIN THE NEURAL NETWORK (NN) ///
        nn_models, nn_losses_dict, _ = get_trained_nn(config, data_preprocessing_info, train_data, val_loader )
        loss_curves(config['nn_model'], config['plotting'], nn_losses_dict)
    
        # /// 7. TRAIN THE PHYSICS-INFORMED NEURAL NETWORK (PINN) ///
        pinn_models, pinn_losses_dict, _ = get_trained_pinn(config, data_preprocessing_info, train_data, val_loader)
        loss_curves(config['pinn_model'], config['plotting'], pinn_losses_dict)

        
        # /// 8. TESTSET RESULTS OF NN, PINN and PROJECTION APPLIED TO BOTH MODELS PREDICTIONS /// 
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
        
        
        #///////////////////////////////////////////////////////////////////////////////////////#
        #///////////////////// 9. ABLATION STUDY (different architectures)//////////////////////#
        #///////////////////////////////////////////////////////////////////////////////////////#
        n_hidden_layers = 2
        options = { 
            'alpha': 1, 
            'patience': 5,
            'n_samples': 1,
            'num_epochs': 1000,
            'hidden_sizes': None,
            'RETRAIN_MODEL': retrain_flag if retrain_flag is not None else config['plotting']['RERUN_ABLATION_STUDY'],
            'n_bootstrap_models': 1, 
            'PRINT_LOSS_VALUES': False,
            'w_matrix': torch.eye(17),
            'APPLY_EARLY_STOPPING': True,
            'activation_fns': ['leaky_relu'] * n_hidden_layers, 
            'extract_results_specific_outputs': ['O2(X)', 'O2(+,X)', 'ne'],
            'output_dir': f'src/ltp_system/figures/Figures_6a/{n_hidden_layers}_hidden_layers/',                          # tables and plots
            'checkpoints_dir': f'output/ltp_system/checkpoints/different_architectures/{n_hidden_layers}_hidden_layers/', # NN weights and biases
            'n_steps': 20,   # N. of different architectures to train
            'dataset_size': 1000,
            'min_neurons_per_layer': 1, 
            'max_neurons_per_layer': 1000,
            'log_random_architectures': True,
            'n_hidden_layers': n_hidden_layers, 
        }        
        df_results_6a_all, df_results_6a_specific, _ = run_ablation_study_architectures(config, large_dataset_path, options)
        Figure_6a_mean_all_outputs(options, df_results_6a_all)
        Figure_6a_specific_outputs(options, df_results_6a_specific, output_labels)
        Figure_computation_times(options, case = 1, file_path=options['output_dir'] + "table_results/computation_times.csv")
        
        # PLOTS OF OUTPUTS AS A FUNCTION OF PRESSURE FOR DIFFERENT POINTS IN FIG. 6A AND FIG. 6D
        target_data_file_path = 'data/ltp_system/const_current/data_50_points_30mA.txt'
        architectures_file_path = options['output_dir'] + "/table_results/architectures.csv"
        run_figures_output_vs_pressure_diff_architectures(config, options, target_data_file_path, architectures_file_path)
        
        
        
        #///////////////////////////////////////////////////////////////////////////////////////#
        #///////////////////////////////// 10. DATA SCALING STUDY //////////////////////////////#
        #///////////////////////////////////////////////////////////////////////////////////////#
        options['n_samples']     = 20
        options['RETRAIN_MODEL'] =  retrain_flag if retrain_flag is not None else config['plotting']['RERUN_DATA_SCALING_STUDY']
        options['hidden_sizes']  = [50,50] 
        options['output_dir']    = 'src/ltp_system/figures/Figures_6d/'
        options['checkpoints_dir'] = f'output/ltp_system/checkpoints/different_datasets/'
        
        # Get the list of files to analyze
        dataset_sizes = [20, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 2500]
        df_results_6e_all, df_results_6e_specific, _ = run_data_scaling_study(config, large_dataset_path, dataset_sizes, options)
        Figure_6e_mean_all_outputs(options, df_results_6e_all)
        Figure_6e_specific_outputs(options, df_results_6e_specific, output_labels)
        Figure_computation_times(options, case = 2, file_path=options['output_dir'] + "table_results/computation_times.csv")

        # PLOTS OF OUTPUTS AS A FUNCTION OF PRESSURE FOR DIFFERENT POINTS IN FIG. 6A AND FIG. 6D
        target_data_file_path = 'data/ltp_system/const_current/data_50_points_30mA.txt'
        run_figures_output_vs_pressure_diff_datasets(config, options, dataset_sizes, target_data_file_path)
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


def preprocess_and_split(config, training_file, dataset_size, data_preprocessing_info):

    # read from the train_path file and randomly select 'dataset_size' rows
    temp_file = select_random_rows(training_file, dataset_size, seed=42)

    # extract dataset from larger file
    _, temp_dataset = load_dataset(config, temp_file)

    # preprocess data with the scalers fitted on larger file
    train_data, _, val_data  = setup_dataset_with_preproprocessing_info(temp_dataset.x, temp_dataset.y, data_preprocessing_info)  

    # create the val loader
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['nn_model']['batch_size'], shuffle=True)

    return train_data, val_loader


def get_trained_nn(config, data_preprocessing_info, train_data, val_loader):
    # create checkpoints directory for the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'nn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # either train the model from scratch ...
    if config['nn_model']['RETRAIN_MODEL']:
        nn_models, nn_losses_dict, _ = get_trained_bootstraped_models(config['nn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
        return nn_models, nn_losses_dict, device
    
    # ... or load the checkpoints
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


def get_trained_pinn(config, data_preprocessing_info, train_data, val_loader):
    # create checkpoints directory for the trained models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'pinn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # either train the model from scratch ...
    if config['pinn_model']['RETRAIN_MODEL']:
        pinn_models, pinn_losses_dict, _ = get_trained_bootstraped_models(config['pinn_model'], config['plotting'], data_preprocessing_info, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data, seed = 'default')
        return pinn_models, pinn_losses_dict, device
    
    # ... or load the checkpoints
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


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                Physics-Consistent Machine Learning Method                  ║
    ║               PART 2: Low-Temperature Plasma System Analysis               ║
    ╚════════════════════════════════════════════════════════════════════════════╝
        
    → Prediction of the steady-state plasma properties
    → 2 scaling studies: ablation study and data scaling
    → Computation of pressure-related trends

    Research Paper: "Physics-consistent machine learning"
    University of Lisbon, Av. Rovisco Pais 1, Lisbon, Portugal.

    """)
    
    print("──────────────────────────────────────────────────────────────────────────────")
    while True:
        print("System Configuration:")
        print("1. Retrain model (Fresh plots and tables)")
        print("2. Use existing model (Load pre-computed results & trained weights)")
        print("3. Define configurations manually using config files")
        
        response = input("\nPlease select configuration (1,2,3): ").strip()
        
        if response == '1':
            # flush the current checkpoints, plots and tables
            retrain = flush_model_artifacts('ltp')
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(retrain)
            break
            
        elif response == '2':
            retrain = False
            print("\n[INFO] Using existing model weights and pre-computed results ...")
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(retrain)
            break

        elif response == '3':
            print("\n[INFO] Using manually defined configurations ...")
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(None)
            break
            
        else:
            print("\n[ERROR] Invalid selection. Please choose 1 (Retrain) or 2 (Use existing).")

