import os
import yaml
import torch
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, Any, Tuple

from src.spring_mass_system.utils import set_seed, get_predicted_trajectory, get_target_trajectory, load_checkpoint, compute_parameters
from src.spring_mass_system.dataset_gen import generate_dataset
from src.spring_mass_system.data_prep import preprocess_data
#from src.spring_mass_system.nn import NeuralNetwork, plot_loss_curves_
from src.spring_mass_system.nn import NeuralNetwork, train_nn, plot_loss_curves_nn, optimize_architecture_nn
from src.spring_mass_system.pinn import PhysicsInformedNN, train_pinn, plot_loss_curves_pinn, optimize_pinn_architecture
from src.spring_mass_system.projection import get_inverse_covariance_matrix, get_projection_df #get_projected_trajectory, 
from src.spring_mass_system.plotter.Figure_2b import plot_predicted_trajectory_vs_target
from src.spring_mass_system.plotter.Figure_2c import plot_predicted_energies_vs_target
from src.spring_mass_system.plotter.Figure_2d import plot_bar_plot
from src.spring_mass_system.plotter.Figure_3 import plot_several_initial_conditions
#from src.spring_mass_system.plotter.Extra_Figure_2 import get_trained_pinn_w_optimal_architecture

# Load the configuration file
def load_config(config_path):
    
    # Load the configuration file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_model = NeuralNetwork(config).to(device)
    loss_fn = nn.MSELoss()
    learning_rate = config['nn_model']['learning_rate']
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

    checkpoint_dir = os.path.join('output', 'spring_mass_system', 'checkpoints', 'nn')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    if config['nn_model']['RETRAIN_MODEL']:
        train_losses, val_losses = train_nn(config, nn_model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir)
        return nn_model, (train_losses, val_losses), device
    else:
        try:
            nn_model, optimizer, _, losses = load_checkpoint(nn_model, optimizer, checkpoint_path)
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
        if not losses['train_losses'] or not losses['val_losses']:
            print("Warning: Loss history not found in checkpoint. Returning empty lists for losses.")
            train_losses, val_losses = [], []

        return nn_model, (losses['train_losses'], losses['val_losses']), device

#   
def get_trained_pinn(config: Dict[str, Any], preprocessed_data: Any, checkpoint_dir) -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pinn_model = PhysicsInformedNN(config).to(device)
    loss_fn = nn.MSELoss()
    learning_rate = config['pinn_model']['learning_rate']
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=learning_rate)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    if config['pinn_model']['RETRAIN_MODEL']:
        losses = train_pinn(config, pinn_model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir)
    else:
        try:
           pinn_model, optimizer, _, losses = load_checkpoint(pinn_model, optimizer, checkpoint_path) 
        except FileNotFoundError:
            raise ValueError("Checkpoint not found. Set RETRAIN_MODEL to True or provide a valid checkpoint.")
        
        if not losses['train_losses'] or not losses['val_losses']:
            print("Warning: Loss history not found in checkpoint. Returning empty lists for losses.")

    return pinn_model, losses

# Plot the loss curves for the NN and PINN models
def plot_loss_curves(config: Dict[str, Any], nn_losses: tuple, pinn_losses: Dict[str, Any]):
    plot_loss_curves_nn(config, *nn_losses)
    plot_loss_curves_pinn(config, pinn_losses)

# retrieves initial conditions from the test loader
def get_inputs_from_loader(loader):
    x_test = np.concatenate([X.cpu().numpy() for X, *_ in loader])

    return x_test


def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # Set seed 
        set_seed(42)

        # Load the configuration file
        config = load_config('configs/spring_mass_system_config.yaml')

        # /// 1. EXTRACT DATASET ///
        df_dataset = load_or_generate_dataset(config)
        
        # /// 2. PREPROCESS THE DATASET ///
        preprocessed_data = preprocess_data(df_dataset)
        test_initial_conditions = get_inputs_from_loader(preprocessed_data['test_loader'])
        
        # /// 3. TRAIN THE NEURAL NETWORK (NN) ///
        set_seed(42)
        nn_model, nn_losses, device_nn = get_trained_nn(config, preprocessed_data)

        # /// 4. TRAIN THE PHYSICS-INFORMED NEURAL NETWORK (PINN) ///
        set_seed(42)
        pinn_model, pinn_losses = get_trained_pinn(config, preprocessed_data, checkpoint_dir=os.path.join('output', 'spring_mass_system', 'checkpoints', 'pinn'))
        
        # /// 5. PLOT LOSS CURVES FOR THE NN AND PINN ///
        #plot_loss_curves(config, nn_losses, pinn_losses)

        # /// 6. EVALUATE ONE INITIAL CONDITION (Fig. 2) ///
        initial_state = [-0.15669316, -2.1848829,   0.09043302, -0.16476968]
        n_time_steps  = 165
        df_target = get_target_trajectory(config, n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_nn   = get_predicted_trajectory(config, preprocessed_data, nn_model,   n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_pinn = get_predicted_trajectory(config, preprocessed_data, pinn_model, n_time_steps = n_time_steps, initial_state = torch.tensor(initial_state))
        df_proj_nn   = get_projection_df(initial_state, n_time_steps, nn_model, torch.eye(4), preprocessed_data, config, df_nn)
        df_proj_pinn = get_projection_df(initial_state, n_time_steps, pinn_model, torch.eye(4), preprocessed_data, config, df_pinn)

        # Make plots
        n_time_steps = 200
        plot_predicted_trajectory_vs_target(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
        #plot_predicted_energies_vs_target(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
        #plot_bar_plot(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn, preprocessed_data)

        # /// 7. EVALUATE SEVERAL INITIAL CONDITIONS (Fig. 3 & Table 1 & Table 2) 
        plot_several_initial_conditions(config, preprocessed_data, nn_model, pinn_model, test_initial_conditions, n_time_steps, N_initial_conditions = 100)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise



if __name__ == "__main__":
    main()


