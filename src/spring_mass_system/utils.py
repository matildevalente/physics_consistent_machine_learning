import os
import torch
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)




def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pgf.preamble": r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc}"
}

plt.rcParams.update(pgf_with_latex)

# I make my own newfig and savefig functions
def newfig(width, nplots = 1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, pad_inches, crop=True):
    try:
        # Save in PDF format
        if crop:
            plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=pad_inches)
        else:
            plt.savefig('{}.pdf'.format(filename))
            
    except Exception as e:
        logging.error(f"An error occurred while saving the file: {e}")
        print(f"Could not save the figure due to an error: {e}")




# Compute the total energy for a given state
def compute_total_energy(config: Dict[str, Any], state: Union[np.ndarray, list, tuple]) -> float:
    """
    Compute the total energy of a spring-mass system.

    Args:
        config (Dict[str, Any]): Configuration dictionary containing system parameters.
        state (Union[np.ndarray, list, tuple]): System state [x1, v1, x2, v2].

    Returns:
        float: Total energy of the system.
    """
    # Extract system parameters
    params = config['spring_mass_system']
    K1, K2 = params['K1'], params['K2']
    M1, M2 = params['M1'], params['M2']
    L1, L2 = params['L1'], params['L2']

    # Unpack state
    x1, v1, x2, v2 = state

    # Compute kinetic and potential energies
    KE = 0.5 * (M1 * v1**2 + M2 * v2**2)
    PE = 0.5 * (K1 * (x1 - L1)**2 + K2 * (x2 - x1 - L2)**2)

    # Return total energy
    return KE + PE


# System dynamics function
def system_dynamics(config, y):

    # Unpack the state vector y into positions (x1, x2) and velocities (v1, v2)
    x1, v1, x2, v2 = y

    # Extract the system's constants from the config file
    K1 = config['spring_mass_system']['K1']
    K2 = config['spring_mass_system']['K2']
    M1 = config['spring_mass_system']['M1']
    M2 = config['spring_mass_system']['M2']
    L1 = config['spring_mass_system']['L1']
    L2 = config['spring_mass_system']['L2']

    # Calculate the acceleration of M1
    dd1 = (-K1 * (x1 - L1) + K2 * (x2 - x1 - L2)) / M1

    # Calculate acceleration of M2
    dd2 = (-K2 * (x2 - x1 - L2)) / M2

    # Return the derivatives
    return np.array([v1, dd1, v2, dd2])


# RK4 integration step function
def rk4_step(config, y, dt):
    f1 = system_dynamics(config, y)
    f2 = system_dynamics(config, y + np.array(f1) * 0.5 * dt)
    f3 = system_dynamics(config, y + np.array(f2) * 0.5 * dt)
    f4 = system_dynamics(config, y + np.array(f3) * dt)
    return y + dt * (np.array(f1) + 2 * np.array(f2) + 2 * np.array(f3) + np.array(f4)) / 6

"""# RK8 integration step function
def rk8_step(config, y, dt):
    k1 = dt * system_dynamics(config, y)
    k2 = dt * system_dynamics(config, y + k1 * 4/27)
    k3 = dt * system_dynamics(config, y + (k1 + k2) / 36)
    k4 = dt * system_dynamics(config, y + (k1 + 3*k3) / 24)
    k5 = dt * system_dynamics(config, y + (5*k1 - 9*k3 + 6*k4) / 8)
    k6 = dt * system_dynamics(config, y + (-11*k1 + 18*k3 - 9*k4 - 2*k5) / 27)
    k7 = dt * system_dynamics(config, y + (17*k1 - 27*k3 + 27*k4 + k5 + k6) / 72)
    k8 = dt * system_dynamics(config, y + (-221*k1 + 981*k3 - 867*k4 + 68*k5 - 33*k6 + 72*k7) / 1080)
    k9 = dt * system_dynamics(config, y + (k1 + 8*k6 + k7) / 90)
    k10 = dt * system_dynamics(config, y + (-1/270 * k1 + 2/27 * k6 + 1/27 * k7 + 2/27 * k8))
    k11 = dt * system_dynamics(config, y + (1/6 * k1 + 2/3 * k7 - 1/3 * k8 + 1/6 * k9))
    k12 = dt * system_dynamics(config, y + (-1/54 * k1 + 1/6 * k7 - 1/18 * k8 - 1/54 * k9 + 1/18 * k10))
    k13 = dt * system_dynamics(config, y + (1/14 * k1 + 1/14 * k9 + 4/7 * k11 - 1/7 * k12))

    return y + (41*k1 + 216*k8 + 27*k9 + 272*k10 + 27*k11 + 216*k12 + 41*k13) / 840"""

# Set seed to garantee reproduc.. of results
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Returns the predicted NN or PINN trajectories in a dataframe format given an initial condition
def get_predicted_trajectory(config, preprocessed_data, model, n_time_steps, initial_state):
    """
    Evaluate the model trajectory from an initial condition and return a DataFrame.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (torch.nn.Module): The trained model.
        n_time_steps (int): Number of time steps to evaluate.
        initial_state (torch.Tensor): Initial state of the system.
        preprocessed_data (Dict[str, Any]): Preprocessed data containing scaler information.

    Returns:
        pd.DataFrame: DataFrame containing the predicted states, energy, and time.
    """
    dt =  config['dataset_generation']['dt_RK'] * config['dataset_generation']['N_RK_STEPS']
    scaler_X = preprocessed_data['scaler_X']

    model.eval()
    device = next(model.parameters()).device

    initial_state = initial_state.to(device)
    current_state = initial_state.clone()

    pred_states = []
    energy_list = []
    time_list = np.arange(0, n_time_steps * dt, dt)

    try:
        with torch.no_grad():
            
            for t in time_list:
                pred_states.append(current_state.cpu().numpy())
                energy = compute_total_energy(config, current_state)
                energy_list.append(energy.item())

                current_state_norm = torch.tensor(scaler_X.transform(current_state.cpu().numpy().reshape(1, -1)), dtype=torch.float32).flatten().to(device)
                
                next_state_norm = model(current_state_norm.unsqueeze(0))
                next_state = torch.tensor(scaler_X.inverse_transform(next_state_norm.cpu().numpy().reshape(1, -1))).flatten().to(device)

                current_state = next_state.clone()

        pred_states_np_array = np.vstack(pred_states)
        df = pd.DataFrame(pred_states_np_array, columns=['x1', 'v1', 'x2', 'v2'])
        df["E"] = energy_list
        df["time(s)"] = time_list

        return df

    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {str(e)}")
        raise


# Returns the target RK trajectory in a dataframe format given an initial condition
def get_target_trajectory(config, n_time_steps, initial_state):
    """
    Evaluate the model trajectory from an initial condition and return a DataFrame.

    Args:
        config (Dict[str, Any]): Configuration dictionary.
        model (torch.nn.Module): The trained model.
        n_time_steps (int): Number of time steps to evaluate.
        initial_state (torch.Tensor): Initial state of the system.
        preprocessed_data (Dict[str, Any]): Preprocessed data containing scaler information.

    Returns:
        pd.DataFrame: DataFrame containing the predicted states, energy, and time.
    """
    
    # To store the evolution of states
    dt_RK = config['dataset_generation']['dt_RK']
    N_RK_STEPS = config['dataset_generation']['N_RK_STEPS']
    dt = dt_RK *  N_RK_STEPS 
    
    #
    energy_list, time_list = [], []
    target_states = torch.tensor([])
    time_index = 0

    # Initial conditions - ensure this is a 1D tensor with exactly 4 elements
    current_state = initial_state.clone().flatten()

    # Compute target states using RK 
    for _ in range(n_time_steps):

        # Append states
        target_states = torch.cat((target_states, current_state.unsqueeze(0)), dim=0)

        # Make predicions ( No need to track gradients )
        for _ in range(N_RK_STEPS):
            with torch.no_grad():  
                # Prediction of RK is the target: only compute one time step, ie, the next state
                next_state = rk4_step(config, current_state, dt_RK)
                current_state = next_state

        # Extract and store output states
        x1, v1, x2, v2 = next_state

        # Compute energy 
        energy = compute_total_energy(config, [x1.item(), v1.item(), x2.item(), v2.item()])
        energy_list.append(energy)

        # Append and update time 
        time_list.append(time_index)
        time_index = time_index + dt

        # Update current state to the next/predicted state
        current_state = next_state.clone()

    # Convert to np arrays
    target_states_np = np.array(target_states)

    # Assuming df_input is your original DataFrame
    df = pd.DataFrame(target_states_np, columns=['x1', 'v1', 'x2', 'v2'])
    df["E"] = energy_list
    df["time(s)"] = time_list
    
    return df


def save_checkpoint(model: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                epoch: int, 
                train_losses: List[float], 
                train_data_losses: List[float],
                train_physics_losses: List[float],
                val_losses: List[float], 
                checkpoint_path: str) -> None:
        
    # Prepare the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_physics_losses': train_physics_losses,
        'train_data_losses': train_data_losses,
        'val_losses': val_losses,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    checkpoint_path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, List[float], List[float]]:
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        train_physics_losses = checkpoint['train_physics_losses']
        train_data_losses = checkpoint['train_data_losses']
        val_losses = checkpoint['val_losses']

        print(f"\nCheckpoint loaded: Epoch {epoch}")
        print(f"Train Losses History: {len(train_losses)} entries")
        print(f"Train Data Losses History: {len(train_physics_losses)} entries")
        print(f"Train Physics Losses History: {len(train_data_losses)} entries")
        print(f"Val Loss History: {len(val_losses)} entries")

        losses = {
        'train_losses': train_losses,
        'train_physics_losses': train_physics_losses,
        'train_data_losses': train_data_losses,
        'val_losses': val_losses
    }

        return model, optimizer, epoch, losses
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        losses = {
            'train_losses': [],
            'train_physics_losses': [],
            'train_data_losses': [],
            'val_losses': []
        }
        return model, optimizer, 0, losses


# Compute the number of NN parameters (weights and biases) based on its architecture
def compute_parameters(layer_config):

    hidden_layers = layer_config[1:-1]
    n_weights = sum(layer_config[i] * layer_config[i+1] for i in range(len(layer_config) - 1)) # x[layer_0] * x[layer_1] + ... + x[layer_N-1]*x[layer_N]
    n_biases = sum(hidden_layers)

    return n_weights + n_biases

