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

from src.ltp_system.utils import set_seed, load_dataset, load_config
from src.ltp_system.data_prep import DataPreprocessor
from src.ltp_system.plotter.eda import apply_eda
from src.ltp_system.pinn_nn import get_trained_bootstraped_models, load_checkpoints, NeuralNetwork
from src.ltp_system.projection import get_inverse_covariance_matrix , print_matrix, compute_projection_results, compute_mape_physical_laws, compute_mape_physical_laws_loki
from src.ltp_system.plotter.loss_curves import loss_curves
from src.ltp_system.plotter.barplots import Figure_4d, Figure_4a, Figure_4b
from src.ltp_system.plotter.Figure_6a import Figure_6a, run_experiment_6a
from src.ltp_system.plotter.Figure_6e import Figure_6e, run_experiment_6e
from src.ltp_system.plotter.outputs_vs_pressure import get_data_Figure_6b, Figure_6b



def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # Set seed 
        set_seed(42)        

        # Load the configuration file
        config = load_config('configs/ltp_system_config.yaml')

        # /// 1. EXTRACT DATASET ///
        df_dataset, full_dataset = load_dataset(config, dataset_dir = 'data/ltp_system/data_1000_points.txt')
        
        # /// 2. PREPROCESS THE DATASET ///
        preprocessed_data = DataPreprocessor(config)
        preprocessed_data.setup_dataset(full_dataset.x, full_dataset.y)        
        #apply_eda(config, preprocessed_data, full_dataset.y)

        # /// 3. TRAIN THE NEURAL NETWORK (NN) ///
        nn_models, nn_losses_dict, device_nn = get_trained_nn(config, preprocessed_data)
        loss_curves(config['nn_model'], config['plotting'], nn_losses_dict)
    
        # /// 4. TRAIN THE PHYSICS-INFORMED NEURAL NETWORK (PINN) ///
        pinn_models, pinn_losses_dict, device_pinn = get_trained_pinn(config, preprocessed_data)
        loss_curves(config['pinn_model'], config['plotting'], pinn_losses_dict)

        
        # /// 5. GET MATRIX TO PERFORM PROJECTION ///
        w_inv_cov = get_inverse_covariance_matrix(config, preprocessed_data, nn_models, device_nn)
        print_matrix(w_inv_cov)

        # /// 6. BARPLOTS COMPARING THE ERROR WHEN APPLYING DIFFERENT SETS OF CONSTRAINTS /// 
        file_path = 'data/ltp_system/data_300_points.txt'
        nn_mape_sem_dict = compute_projection_results(config['nn_model'], torch.eye(17), file_path, preprocessed_data, nn_models)
        pinn_mape_sem_dict = compute_projection_results(config['pinn_model'], torch.eye(17), file_path, preprocessed_data, pinn_models)
        Figure_4a(config['plotting'], nn_mape_sem_dict, pinn_mape_sem_dict)
        Figure_4d(nn_mape_sem_dict, config['nn_model'], config['plotting'])  

        # /// 7. BARPLOTS COMPARING THE ERROR IN COMPLIANCE WITH PHYSICAL LAWS /// 
        laws_dict = get_laws_dict(file_path, preprocessed_data, nn_models, pinn_models)
        Figure_4b(config['plotting'], laws_dict)
    
        # /// 8. CONST CURRENT PLOTS ///
        #file_path = 'data/ltp_system/const_current/data_50_points_30mA.txt'
        #preds_dict, errors_dict = get_data_Figure_6b(config, nn_models, file_path, torch.eye(17))
        #Figure_6b(config, preds_dict, errors_dict)

        # /// 9. NN and NN_Proj errors as a func of the dataset size + approximate loki computation time /// 
        options = {
            'n_bootstrap_models': 10, 
            'hidden_sizes': [451, 315, 498, 262],
            'num_epochs':  100,
            'activation_fns': ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu'], 
            'w_matrix': torch.eye(17), 
            'PRINT_LOSS_VALUES': False,
            'RETRAIN_MODEL': False, 
        }
        # Get the list of files to analyze
        dataset_sizes = [175, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]
        large_dataset = 'data/ltp_system/data_15000_points.txt'
        df_results_6e = run_experiment_6e(config, large_dataset, dataset_sizes, options)
        Figure_6e(config['plotting'], df_results_6e)

        # /// 10. NN and NN_Proj errors as a func of the model's parameters + NN training time /// 
        """options = {
            'n_bootstrap_models':  10,
            'activation_func': 'leaky_relu', 
            'max_hidden_layers': 1, 
            'min_hidden_layers':1,
            'max_neurons_per_layer': 75, 
            'min_neurons_per_layer': 1, 
            'n_steps': 20,
            'w_matrix': torch.eye(17), 
            'num_epochs': 50,
            'PRINT_LOSS_VALUES': False,
            'RETRAIN_MODEL': False
        }
        file_path = 'data/ltp_system/data_600_points.txt'
        df_results_6a = run_experiment_6a(config, file_path, options)
        Figure_6a(config['plotting'], df_results_6a)"""




        """# /// 6. EVALUATE ONE INITIAL CONDITION (Fig. 2) ///
        initial_state = [ 0.23334019,  0.45744362, -0.07293496, -0.43316936] #[0.37987354, 0.19292271, 0.43858615, 0.82498616]##0.9396753206665032,2.6248020727512547,0.679625428401559,-1.5368544850961408
        n_time_steps = 50
        df_target = get_target_trajectory(config, n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_nn   = get_predicted_trajectory(config, preprocessed_data, nn_model,   n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_pinn = get_predicted_trajectory(config, preprocessed_data, pinn_model, n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_proj_nn   = get_projection_df(initial_state, n_time_steps, nn_model, torch.eye(4), preprocessed_data, config, df_nn)
        df_proj_pinn = get_projection_df(initial_state, n_time_steps, pinn_model, torch.eye(4), preprocessed_data, config, df_pinn)

        # Make plots
        plot_predicted_energies_vs_target(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
        plot_predicted_trajectory_vs_target(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
        plot_bar_plot(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn, preprocessed_data)

        # /// 6. EVALUATE SEVERAL INITIAL CONDITIONS (Fig. 3 & Table 1) ///test_initial_conditions
        plot_several_initial_conditions(config, preprocessed_data, nn_model, pinn_model, test_initial_conditions, w_inv_cov, n_time_steps = n_time_steps)"""


    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


def get_trained_nn(config: Dict[str, Any], preprocessed_data: Any) -> Tuple[nn.Module, Tuple[list, list]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'nn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    train_data = preprocessed_data.train_data
    val_loader = torch.utils.data.DataLoader(preprocessed_data.val_data, batch_size=config['nn_model']['batch_size'], shuffle=True)
    
    if config['nn_model']['RETRAIN_MODEL']:

        nn_models, nn_losses_dict, _ = get_trained_bootstraped_models(config['nn_model'], config['plotting'], preprocessed_data, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data)
        return nn_models, nn_losses_dict, device
    else:
        try:
            nn_models, nn_losses_dict, _, _, _ = load_checkpoints(config['nn_model'], NeuralNetwork, checkpoint_dir)

            # Check if any losses were loaded
            if (nn_losses_dict['losses_train_mean'].size == 0 or 
                nn_losses_dict['losses_val_mean'].size == 0 or 
                nn_losses_dict['epoch'].size == 0):
                print("Warning: Incomplete loss history found in checkpoint. Retraining the model.")
                nn_models, nn_losses_dict, _ = get_trained_bootstraped_models(config['nn_model'], config['plotting'], preprocessed_data, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data)
            else:
                print("Checkpoint loaded successfully with loss history.")

        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")

    return nn_models, nn_losses_dict, device


def get_trained_pinn(config: Dict[str, Any], preprocessed_data: Any) -> Tuple[nn.Module, Tuple[list, list]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.join('output', 'ltp_system', 'checkpoints', 'pinn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    train_data = preprocessed_data.train_data
    val_loader = torch.utils.data.DataLoader(preprocessed_data.val_data, batch_size=config['pinn_model']['batch_size'], shuffle=True)
    
    if config['pinn_model']['RETRAIN_MODEL']:

        pinn_models, pinn_losses_dict, _ = get_trained_bootstraped_models(config['pinn_model'], config['plotting'], preprocessed_data, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data)
        return pinn_models, pinn_losses_dict, device
    else:
        try:
            pinn_models, pinn_losses_dict, _, _, _ = load_checkpoints(config['pinn_model'], NeuralNetwork, checkpoint_dir)
        
            # Check if any losses were loaded
            if (pinn_losses_dict['losses_train_mean'].size == 0 or 
                pinn_losses_dict['losses_val_mean'].size == 0 or 
                pinn_losses_dict['epoch'].size == 0):
                print("Warning: Incomplete loss history found in checkpoint. Retraining the model.")
                pinn_models, pinn_losses_dict, _ = get_trained_bootstraped_models(config['pinn_model'], config['plotting'], preprocessed_data, nn.MSELoss(), checkpoint_dir, device, val_loader, train_data)
            else:
                print("Checkpoint loaded successfully with loss history.")     
        
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
    return pinn_models, pinn_losses_dict, device


def get_laws_dict(file_name, preprocessed_data, nn_models, pinn_models):
    nn_laws_dict   = compute_mape_physical_laws(file_name, preprocessed_data, nn_models, torch.eye(17), "nn_model")
    pinn_laws_dict = compute_mape_physical_laws(file_name, preprocessed_data, pinn_models, torch.eye(17), "pinn_model")
    loki_laws_dict = compute_mape_physical_laws_loki(file_name, preprocessed_data, torch.eye(17))

    laws_dict = {**nn_laws_dict, **pinn_laws_dict, **loki_laws_dict}

    keys = list(nn_laws_dict['nn_model'].keys())
    
    # Save results to local directory
    output_dir = 'output/ltp_system/'
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'physical_compliance_errors.csv')

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










