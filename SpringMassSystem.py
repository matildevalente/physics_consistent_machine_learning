import os
import yaml
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Dict, Any, Tuple

from clean import flush_model_artifacts, flush_data, flush_lambda_study
from src.spring_mass_system.utils import set_seed, get_predicted_trajectory, get_target_trajectory, load_checkpoint
from src.spring_mass_system.dataset_gen import generate_dataset
from src.spring_mass_system.data_prep import preprocess_data
from src.spring_mass_system.pinn_nn import NeuralNetwork, train_model, plot_loss_curves_
from src.spring_mass_system.projection import get_projection_df 
from src.spring_mass_system.plotter.Figure_2b import plot_predicted_trajectory_vs_target
from src.spring_mass_system.plotter.Figure_2c import plot_predicted_energies_vs_target
from src.spring_mass_system.plotter.Figure_2d import plot_bar_plot
from src.spring_mass_system.plotter.Figure_3 import plot_several_initial_conditions
from src.spring_mass_system.plotter.Extra_Figure_1 import plot_pinn_errors_vs_lambda

# Load the configuration file
def load_config(retrain_flag, regenerate_data_flag, rerun_lambda_study, config_path):
    
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        # overwrite the manually defined retraining configurations with the user input
        if(retrain_flag is not None):
            # if the dataset was flushed
            config['dataset_generation']['GENERATE_DATASET'] = regenerate_data_flag

            # if the checkpoints and results were flushed
            config['nn_model']['RETRAIN_MODEL'] = retrain_flag
            config['pinn_model']['RETRAIN_MODEL'] = retrain_flag
            config['fig_3_options']['rerun_results'] = retrain_flag

            # if the results from the lambda_physics analysis were flushed
            config['pinn_model']['RUN_LAMBDA_STUDY'] = rerun_lambda_study

        return config


# Load the dataset from a .csv file or generate the entire dataset
def load_or_generate_dataset(config):
    # Define column names
    column_names = ['x1', 'v1', 'x2', 'v2', 'x1(t+dt)', 'v1(t+dt)', 'x2(t+dt)', 'v2(t+dt)']

    GENERATE_DATASET = config['dataset_generation']['GENERATE_DATASET']

    if GENERATE_DATASET:
        return generate_dataset(config, column_names)
    elif GENERATE_DATASET is False:
        try:
            return pd.read_csv('data/spring_mass_system/data.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Dataset file not found. Please generate the dataset or provide the correct file path.")
    else:
        raise ValueError("Invalid value for GENERATE_DATASET in config. Must be True or False.")


# 
def get_trained_nn(config: Dict[str, Any], preprocessed_data: Any) -> Tuple[nn.Module, Tuple[list, list]]:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_model = NeuralNetwork(config['nn_model']).to(device)
    learning_rate = config['nn_model']['learning_rate']
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

    checkpoint_dir = os.path.join('output', 'spring_mass_system', 'checkpoints', 'nn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    if config['nn_model']['RETRAIN_MODEL']:
        losses = train_model(config, config['nn_model'], nn_model, preprocessed_data, optimizer, device, checkpoint_dir)
    else:
        try:
            nn_model, optimizer, _, losses = load_checkpoint(nn_model, optimizer, checkpoint_path)
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
        if not losses['train_losses'] or not losses['val_losses']:
            losses = []
            print("Warning: Loss history not found in checkpoint. Returning empty lists for losses.")

    return nn_model, losses


#   
def get_trained_pinn(config: Dict[str, Any], preprocessed_data: Any, checkpoint_dir) -> tuple:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pinn_model = NeuralNetwork(config['pinn_model']).to(device)
    learning_rate = config['pinn_model']['learning_rate']
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=learning_rate)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    if config['pinn_model']['RETRAIN_MODEL']:
        losses = train_model(config,config['pinn_model'], pinn_model, preprocessed_data, optimizer, device, checkpoint_dir)
    else:
        try:
           pinn_model, optimizer, _, losses = load_checkpoint(pinn_model, optimizer, checkpoint_path) 
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
        if not losses['train_losses'] or not losses['val_losses']:
            losses = []
            print("Warning: Loss history not found in checkpoint. Returning empty lists for losses.")
    
    return pinn_model, losses


# Plot the loss curves for the NN and PINN models
def plot_loss_curves(config: Dict[str, Any], nn_losses: Dict[str, Any], pinn_losses: Dict[str, Any]):
    plot_loss_curves_(config, nn_losses, model_name = "nn")
    plot_loss_curves_(config, pinn_losses, model_name = "pinn")


# retrieves initial conditions from the test loader
def get_inputs_from_loader(loader):
    x_test = np.concatenate([X.cpu().numpy() for X, *_ in loader])

    return x_test


def main(retrain_flag, regenerate_data_flag, rerun_lambda_study):
    try:
        # /// 1. SETUP LOGGING ///
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # /// 2. SET SEED ///
        set_seed(42)

        # /// 3. LOAD CONFIGURATION FILE /// 
        config = load_config(retrain_flag, regenerate_data_flag, rerun_lambda_study, 'configs/spring_mass_system_config.yaml')

        # /// 4. EXTRACT DATASET ///
        df_dataset = load_or_generate_dataset(config)
        
        # /// 5. PREPROCESS THE DATASET ///
        preprocessed_data = preprocess_data(df_dataset)
        test_initial_conditions = get_inputs_from_loader(preprocessed_data['test_loader'])
        
        # /// 6. TRAIN THE NEURAL NETWORK (NN) ///
        nn_model, nn_losses = get_trained_nn(config, preprocessed_data)

        # /// 7. TRAIN THE PHYSICS-INFORMED NEURAL NETWORK (PINN) ///
        checkpoint_dir=os.path.join('output', 'spring_mass_system', 'checkpoints', 'pinn')
        plot_pinn_errors_vs_lambda(config, preprocessed_data, N_lambdas = 20)
        pinn_model, pinn_losses = get_trained_pinn(config, preprocessed_data, checkpoint_dir)
        
        # /// 8. PLOT LOSS CURVES FOR THE NN AND PINN ///
        plot_loss_curves(config, nn_losses, pinn_losses)

        # /// 9. EVALUATE ONE INITIAL CONDITION (Fig. 2) ///
        initial_state = [-0.16, -2.18,   0.09, -0.16]
        n_time_steps  = 165
        df_target = get_target_trajectory(config, n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_nn   = get_predicted_trajectory(config, preprocessed_data, nn_model,   n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_pinn = get_predicted_trajectory(config, preprocessed_data, pinn_model, n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_proj_nn   = get_projection_df(initial_state, n_time_steps, nn_model, torch.eye(4), preprocessed_data, config, df_nn)
        df_proj_pinn = get_projection_df(initial_state, n_time_steps, pinn_model, torch.eye(4), preprocessed_data, config, df_pinn)

        # Make plots
        plot_predicted_trajectory_vs_target(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
        plot_predicted_energies_vs_target(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
        plot_bar_plot(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn, preprocessed_data)

        # /// 10. EVALUATE SEVERAL INITIAL CONDITIONS (Fig. 3 & Table 1 & Table 2) 
        n_time_steps = 200
        plot_several_initial_conditions(config, preprocessed_data, nn_model, pinn_model, test_initial_conditions, n_time_steps, N_initial_conditions = 100)
        

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                Physics-Consistent Machine Learning Method                  ║
    ║                 PART 1: Spring-mass System Analysis                        ║
    ╚════════════════════════════════════════════════════════════════════════════╝
        
    → Data generation
    → Comparative analysis of the models performance for a given initial condition
    → Comparative analysis of the models performance for several initial conditions

    Research Paper: "Physics-consistent machine learning"
    University of Lisbon, Av. Rovisco Pais 1, Lisbon, Portugal.

    """)
    
    print("──────────────────────────────────────────────────────────────────────────────")
    while True:
        print("System Configuration:")
        print("1. Regenerate dataset and retrain model (Fresh plots and tables)")
        print("2. Only retrain model (Fresh plots and tables)")
        print("3. Use existing model (Load pre-computed results & trained weights)")
        print("4. Define configurations manually using config files")

        response = input("\nPlease select configuration (1,2,3,4): ").strip()

        if response == '1':
            # flush dataset and the current checkpoints, plots and tables 
            regenerate_data = flush_data()
            retrain = flush_model_artifacts('spring')
            rerun_lambda_study = flush_lambda_study('spring')
            
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(retrain, regenerate_data, rerun_lambda_study)
            break
        
        elif response == '2':
            regenerate_data = False
            # flush the current checkpoints, plots and tables
            retrain = flush_model_artifacts('spring')
            rerun_lambda_study = flush_lambda_study('spring')
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(retrain, regenerate_data, rerun_lambda_study)
            break
            
        elif response == '3':
            regenerate_data = False
            retrain = False
            rerun_lambda_study = False
            print("\n[INFO] Using existing model weights and pre-computed results ...")
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(retrain, regenerate_data, rerun_lambda_study)
            break

        elif response == '4':
            print("\n[INFO] Using manually defined configurations ...")
            print("──────────────────────────────────────────────────────────────────────────────\n")
            main(None)
            break
            
        else:
            print("\n[ERROR] Invalid selection. Please choose 1 (Retrain) or 2 (Use existing).")


