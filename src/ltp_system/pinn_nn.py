import os
import sys
import math
import copy
import time
import torch 
import random
import numpy as np
import casadi as ca
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from io import StringIO
from itertools import cycle
import torch.optim as optim
device = torch.device("cpu")
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.ltp_system.utils import set_seed


# --------------------------------------------------------------------------- NN and PINN Class
# This class is used for the NN and PINN. For the NN training the lambda_physics
# ie the weight of the physics constraint in the loss function is set to 0
class NeuralNetwork(nn.Module):
    def __init__(self, config_model):
        super(NeuralNetwork, self).__init__()
        self.hidden_size_arr = config_model['hidden_sizes']
        #self.activation_functions = config_model['activation_fns']
        self.activation_functions = self._parse_activation_functions(config_model['activation_fns'])
        self.lr = config_model['learning_rate']
        self.batch_size = config_model['batch_size']
        self.lambda_physics = config_model['lambda_physics']
        
        layers = []
        layers.append(nn.Linear(3, self.hidden_size_arr[0]))
        for i in range(len(self.hidden_size_arr) - 1):
            layers.append(nn.Linear(self.hidden_size_arr[i], self.hidden_size_arr[i + 1]))

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(self.hidden_size_arr[-1], 17)

    def forward(self, x):
        if x.dtype != torch.float64:
            x = x.double()

        for i, layer in enumerate(self.hidden_layers):
            x = self.activation_functions[i](layer(x))
        x = self.output_layer(x)
        return x
    
    def _parse_activation_functions(self, activation_strings):
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'elu': F.elu,
            'selu': F.selu,
        }
        
        return [activation_map.get(act.lower(), F.relu) for act in activation_strings]

    """def optimize_architecture(self, config_model, val_loader, train_loader, preprocessed_data):

        def get_random_architectures_(config_model):
            hidden_size_arr = []
            base_architecture = [10, 10,10]
            num_arquitectures = config_model['num_arquitectures']
            
            # Randomly generate NN architectures 
            for _ in range(num_arquitectures):
                num_positions = random.randint(3, 7)  
                new_architecture = [random.randint(3, 500) for _ in range(num_positions)]
            
                if base_architecture not in hidden_size_arr:
                    hidden_size_arr.append(base_architecture)
                
                hidden_size_arr.append(new_architecture)

            # Sort architectures by the total number of neurons (complexity)
            hidden_size_arr = sorted(hidden_size_arr, key=lambda x: sum(x))

            return hidden_size_arr

        def generate_config_(config_model, hidden_sizes, activation_fn):
            return {
                'RETRAIN_MODEL': config_model['RETRAIN_MODEL'],
                'input_size': config_model['input_size'],
                'hidden_sizes': hidden_sizes,
                'output_size': config_model['output_size'],
                'num_epochs': config_model['num_epochs'],
                'learning_rate': config_model['learning_rate'],
                'activation_fn': activation_fn,
                'loss_physics_weight': config_model['loss_physics_weight'],
            }
            
        random_architectures = get_random_architectures_(config_model)

        df_ = pd.DataFrame(columns=['architecture', 'activ_func', 'val_loss'])
        rows = []

        for architecture in random_architectures:

            set_seed(42)

            # get activation functions
            activation_fns = random.choices(self.activation_functions, k=len(architecture))    

            # Create a new instance of the neural network
            config_temp = generate_config_(config_model, architecture, activation_fns)
            model = PhysicsAwareNet(config_temp).to(device)
            model.to(torch.double) 

            # Define the loss function and optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=config_temp['learning_rate'])
            
            # 4. PLATEAU BASED STOPPING - Define the scheduler and monitor validation loss
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
                
            # 5. Train Network Without Physical Bias
            _, _, _, val_losses, _, _ = train_PINN(model, optimizer, train_loader, val_loader, scheduler, preprocessed_data)

            new_row = {
                "architecture": architecture,
                "activ_func": activation_fns,
                "val_loss": val_losses[-1]
            }
            rows.append(new_row)

        df_ = pd.DataFrame(rows, columns=['architecture', 'activ_func', 'val_loss'])

        # Find the index of the minimum val_loss
        min_loss_index = df_['val_loss'].idxmin()
        best_row = df_.loc[min_loss_index]

        return best_row"""

    def predict(self, x):
        # call the forward method to generate predictions
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


# --------------------------------------------------------------------------- Methods for PINN Training & Validation
# Compute the mean and std of the losses across the bootstraped models
def aggregate_losses_(losses_train, losses_train_physics, losses_train_data, losses_val):
    
    # Find max length
    max_epochs = max(len(sublist) for sublist in losses_train)
    num_models = len(losses_train)

    mean_losses, std_losses = [], []

    for losses in [losses_train, losses_train_physics, losses_train_data, losses_val]:

        losses_padded = [sublist + [np.nan] * (max_epochs - len(sublist)) for sublist in losses]
        
        # Convert to numpy array
        losses_padded_arr = np.array(losses_padded)
        
        # Compute mean and std across columns, ignoring NaN values
        mean_values = np.nanmean(losses_padded_arr, axis=0)
        std_values = np.nanstd(losses_padded_arr, axis=0) / np.sqrt(num_models)

        mean_losses.append(mean_values)
        std_losses.append(std_values)

    losses_dict = {
        'epoch': np.arange(1, max_epochs + 1),
        'losses_train_mean': mean_losses[0],
        'losses_train_err': std_losses[0],
        'losses_train_physics_mean': mean_losses[1],
        'losses_train_physics_err': std_losses[1],
        'losses_train_data_mean': mean_losses[2],
        'losses_train_data_err': std_losses[2],
        'losses_val_mean': mean_losses[3],
        'losses_val_err': std_losses[3]
    }

    # Return dictionary with aggregated losses    
    return losses_dict

# Loop over each model and train all the bootstraped models
def get_trained_bootstraped_models(config_model, config_plotting, preprocessed_data, loss_fn, checkpoint_dir, device, val_loader, train_data, seed):
    
    if config_model['lambda_physics'] == [0,0,0]:
        model_name = "NN"
    else:
        model_name = "PINN"

    n_bootstrap_models = config_model['n_bootstrap_models']

    # Plot error as a function of seed
    models_list = []
    losses_train_physics, losses_train_data, losses_train_total, losses_val = [], [], [], []

    print(f"Training Dataset Size: {len(train_data)}")

    start_time = time.time()
  
    for idx in tqdm(range(n_bootstrap_models), desc=f"Training Bootstraped {model_name}"):
        # Each bootstraped model is initialized w a different seed
        if(seed == 'default'):
            set_seed(idx + 1)
        else:
            set_seed(seed)
        
        # Create a new instance of the neural network
        model = NeuralNetwork(config_model).to(device)
        model.to(torch.double) 

        # Define the loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=config_model['learning_rate'])
        
        # Create a bootstrap sample for training
        bootstrap_indices = random.choices(range(len(train_data)), k=len(train_data))
        bootstrap_train_data = torch.utils.data.Subset(train_data, bootstrap_indices)

        # 4. PLATEAU BASED STOPPING - Define the scheduler and monitor validation loss
        train_loader = torch.utils.data.DataLoader(bootstrap_train_data, batch_size=config_model['batch_size'], shuffle=True) 
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        # 5. Train Network Without Physical Bias
        model_losses_dict = train_model(config_plotting, config_model, model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir, scheduler, train_loader, val_loader)
        models_list.append(model)

        # 6. 
        losses_train_total.append(model_losses_dict['train_losses'])
        losses_train_physics.append(model_losses_dict['train_physics_losses'])
        losses_train_data.append(model_losses_dict['train_data_losses'])
        losses_val.append(model_losses_dict['val_losses'])

    end_time = time.time()
    training_time = end_time - start_time
    losses_dict_aggregated = aggregate_losses_(losses_train_total, losses_train_physics, losses_train_data, losses_val)
    save_checkpoints(models_list, losses_dict_aggregated, checkpoint_dir, config_model, training_time )

    return models_list, losses_dict_aggregated, training_time

def train_model(config_plotting, config_model, model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir, scheduler, train_loader, val_loader, print_every=10):
    model.to(device)
    train_losses, train_physics_losses, train_data_losses, val_losses = [], [], [], []
    
    num_epochs = config_model['num_epochs']
    best_model_state = None
    best_model_epoch = 0
    
    # Ensure the directory exists
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    for epoch in range(0, num_epochs):
        # --------------------------- Training loop
        model.train()  # set mode
        train_loss_dict = _run_epoch_train(config_model, model, train_loader, loss_fn, optimizer, device, preprocessed_data)
        # append training loss values
        train_losses.append(train_loss_dict['train_loss'])
        train_physics_losses.append(train_loss_dict['train_weighted_physics_loss'])
        train_data_losses.append(train_loss_dict['train_weighted_data_loss'])

        # --------------------------- Validation loop
        model.eval()  # set mode
        with torch.no_grad():
            val_loss = _run_epoch_val(model, val_loader, loss_fn, device, scheduler)
            val_losses.append(val_loss)
        
        # --------------------------- Print loss as a func of epochs
        if (config_plotting['PRINT_LOSS_VALUES'] == True):
            if epoch % print_every == 0:
                _print_epoch_summary(epoch, train_loss_dict, val_loss)
        
        # --------------------------- Save current model state if we're at patience interval
        if len(val_losses) % config_model['patience'] == 0:
            best_model_state = copy.deepcopy(model.state_dict())
            best_model_epoch = epoch
        
        # --------------------------- Check if the convergence criterion is met & early stopping
        if(config_model['APPLY_EARLY_STOPPING']):
            if stop_training(val_losses, train_losses, config_model['patience'], config_model['alpha']):
                if(config_plotting['PRINT_LOSS_VALUES']):
                    print(f"Stopping training as stopping criterion is met.")
                break
    
    return {
        'train_losses': train_losses,
        'train_physics_losses': train_physics_losses,
        'train_data_losses': train_data_losses,
        'val_losses': val_losses,
        'best_epoch': best_model_epoch
    }

# --------------------------------------------------------------- Residual computation
# Define discharge current physical constraints
def get_current_law_residual(x_norm, y_pred_norm, preprocessed_data):
    
    # 1. get input current normalized
    I_model = x_norm[:,1]

    # 2. revert normalization in the model predictions
    x, y_pred   = preprocessed_data.inverse_transform(x_norm, y_pred_norm)

    # 3. compute current using model's outputs
    R = x[:,2]
    ne = y_pred[:,16]
    vd = y_pred[:,14]
    e = 1.602176634e-19
    pi = np.pi
    I_calc = e * ne * vd * pi * R * R

    # 4. apply normalization on the computed current using model's outputs
    if 1 in preprocessed_data.skewed_features_in:
        I_calc = torch.log1p(torch.tensor(I_calc))
    I_calc = I_calc.reshape(-1, 1)
    I_calc = torch.from_numpy((preprocessed_data.scalers_input[1]).transform(I_calc).reshape(-1, 1))

    # 5. return the residual of the normalized current
    return I_model - I_calc


def get_pressure_law_residual(x_norm, y_pred_norm, preprocessed_data):

    # 1. get input pressure normalized
    P_model = x_norm[:,0]

    # 2. revert normalization in the model predictions
    x, y_pred   = preprocessed_data.inverse_transform(x_norm, y_pred_norm)

    # 3. compute pressure using model's outputs
    num_species = 11
    Tg = y_pred[:,11]
    k_b = 1.380649E-23
    concentrations = 0
    for species_idx in range(num_species):
        concentrations += y_pred[:,species_idx]
    P_calc = concentrations * Tg * k_b

    # 4. apply normalization on the computed pressure using model's outputs
    if 0 in preprocessed_data.skewed_features_in:
        P_calc = torch.log1p(torch.tensor(P_calc))
    P_calc = P_calc.reshape(-1, 1)
    P_calc = torch.from_numpy((preprocessed_data.scalers_input[0]).transform(P_calc).reshape(-1, 1))

    return P_model - P_calc


def get_quasi_neutrality_law_residual(x_norm, y_pred_norm, preprocessed_data):

    # 1. get predicted electron density normalized
    ne_model = y_pred_norm[:,16]

    # 2. revert normalization in the model predictions
    x, y_pred   = preprocessed_data.inverse_transform(x_norm, y_pred_norm)

    # 3. compute ne using model's outputs
    ne_calc = y_pred[:,4] + y_pred[:,7] - y_pred[:,8]         # ne = O2(+,X) + O(+,gnd) - O(-,gnd)

    # 4. apply normalization to the computed ne 
    if 16 in preprocessed_data.skewed_features_out:
        ne_calc = torch.log1p(torch.tensor(ne_calc))
    ne_calc = ne_calc.reshape(-1, 1)
    ne_calc = torch.from_numpy((preprocessed_data.scalers_output[16]).transform(ne_calc).reshape(-1, 1))

    return ne_model - ne_calc
    

def _compute_pinn_loss(config_model, input_norm, y_pred_norm, y_target_norm, preprocessed_data, loss_fn):

    p_residual  = get_pressure_law_residual(input_norm, y_pred_norm, preprocessed_data)
    i_residual  = get_current_law_residual(input_norm, y_pred_norm, preprocessed_data)
    ne_residual = get_quasi_neutrality_law_residual(input_norm, y_pred_norm, preprocessed_data)

    # compute loss for each constraint
    P_constraint = torch.mean((p_residual) ** 2)
    I_contraint = torch.mean((i_residual) ** 2)
    ne_contraint = torch.mean((ne_residual) ** 2)

    # Compute MSE of energy conservation
    physics_weights = config_model['lambda_physics']
    loss_physics_weighted = physics_weights[0] * P_constraint + physics_weights[1] * I_contraint + physics_weights[2] * ne_contraint

    # compute loss data
    loss_data_weighted = (1 - (physics_weights[0] + physics_weights[1] + physics_weights[2])) * loss_fn(y_pred_norm, y_target_norm)
    
    # compute weighted loss 
    loss_total_pinn = loss_physics_weighted + loss_data_weighted

    return {
        'loss_physics_weighted': loss_physics_weighted,
        'loss_data_weighted': loss_data_weighted,
        'loss_total_pinn': loss_total_pinn
    }

# --------------------------------------------------------------- Training loop
def _run_epoch_train(config_model, model, train_loader, loss_fn, optimizer, device, preprocessed_data):
    epoch_loss_total = 0.0
    epoch_loss_physics = 0.0
    epoch_loss_data = 0.0
    
    # for each epoch, rain the training loop
    for (batch_idx, batch) in enumerate(train_loader):
        (inputs, targets) = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss_dict = _compute_pinn_loss(config_model, inputs, outputs, targets, preprocessed_data, loss_fn)           
        loss_dict['loss_total_pinn'].backward()
        optimizer.step()
        
        epoch_loss_total += loss_dict['loss_total_pinn'].item() 
        epoch_loss_physics += loss_dict['loss_physics_weighted'].item()
        epoch_loss_data += loss_dict['loss_data_weighted'].item()
    
    n_batches = len(train_loader)
    epoch_loss_total   = epoch_loss_total / n_batches
    epoch_loss_physics = epoch_loss_physics / n_batches
    epoch_loss_data    = epoch_loss_data / n_batches
    
    return {
        'train_loss': epoch_loss_total,
        'train_weighted_physics_loss': epoch_loss_physics,
        'train_weighted_data_loss': epoch_loss_data
    }

# --------------------------------------------------------------- Validation loop
def _run_epoch_val(model, val_loader, loss_fn, device, scheduler):

    total_val_loss = 0
    # Calculate Validation Loss
    for (batch_idx, batch) in enumerate(val_loader):
        # (predictors, targets)
        (inputs, targets) = batch                        
        inputs, targets = inputs.to(device), targets.to(device)
        
        # compute outputs
        outputs = model(inputs)
        loss_val = loss_fn(outputs, targets)

        # accumulate avgs
        total_val_loss += loss_val.item()                 
    
    n_batches = len(val_loader)
    total_val_loss = total_val_loss/n_batches

    # Reduce Learning Rate When Validation Loss Plateaus
    scheduler.step(total_val_loss)

    return total_val_loss

# --------------------------------------------------------------- Print loss as a func of epochs
def _print_epoch_summary(epoch, train_loss_dict, val_loss):
    def _percentage(part, whole):
        return (part / whole) * 100
    
    print(
        f'epoch = {epoch:5d}   '
        f'L_train = {train_loss_dict["train_loss"]:.2e}   '
        f'L_train_data = {train_loss_dict["train_weighted_data_loss"]:.2e} '
        f'({_percentage(train_loss_dict["train_weighted_data_loss"], train_loss_dict["train_loss"]):.2f}%)   '
        f'L_train_physics = {train_loss_dict["train_weighted_physics_loss"]:.2e} '
        f'({_percentage(train_loss_dict["train_weighted_physics_loss"], train_loss_dict["train_loss"]):.2f}%)   '
        f'L_val= {val_loss:.2e}'
    )



# --------------------------------------------------------------------------- Save and load checkpoints of the num_models from a local directory
def save_checkpoints(models, losses_dict, save_dir, config_model, training_time):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the aggregated losses separately
    aggregated_losses = {
        'epoch': losses_dict['epoch'],
        'losses_train_mean': losses_dict['losses_train_mean'], 
        'losses_train_err': losses_dict['losses_train_err'], 
        'losses_train_physics_mean': losses_dict['losses_train_physics_mean'], 
        'losses_train_physics_err': losses_dict['losses_train_physics_err'], 
        'losses_train_data_mean': losses_dict['losses_train_data_mean'], 
        'losses_train_data_err': losses_dict['losses_train_data_err'], 
        'losses_val_mean': losses_dict['losses_val_mean'], 
        'losses_val_err': losses_dict['losses_val_err'], 
    }
    
    losses_path = os.path.join(save_dir, 'aggregated_losses.pth')
    torch.save(aggregated_losses, losses_path)
    print("Aggregated losses saved.")

    ################ save the architecture, activation functions, and training time of the aggregated model
    aggregated_model_info = {
        'hidden_sizes': config_model['hidden_sizes'], 
        'activation_fns': config_model['activation_fns'],  
        'training_time': training_time
    }
    path_1 = os.path.join(save_dir, 'aggregated_model_info.pth')
    torch.save(aggregated_model_info, path_1)

    # Save each model with hidden sizes and activation functions
    for i, model in enumerate(models):
        checkpoint_2 = {
            'model_state_dict': model.state_dict(),
        }
        checkpoint_path = os.path.join(save_dir, f'model_{i}_checkpoint.pth')
        torch.save(checkpoint_2, checkpoint_path)
        print(f"Model {i} saved with hidden sizes and activation function.")

    print(f"All checkpoints and aggregated losses saved in {save_dir}")

def load_checkpoints(config_model, model_class, save_dir):
    loaded_models = []
    num_models = config_model['n_bootstrap_models']

    # Load aggregated losses
    losses_path = os.path.join(save_dir, 'aggregated_losses.pth')
    if os.path.exists(losses_path):
        loaded_losses = torch.load(losses_path)
    else:
        print("Warning: Aggregated losses file not found.")
        loaded_losses = {}

    # Load aggregated model training info
    path_1 = os.path.join(save_dir, 'aggregated_model_info.pth')
    if os.path.exists(path_1):
        checkpoint_1 = torch.load(path_1)
        hidden_sizes   = checkpoint_1['hidden_sizes']
        activation_fns = checkpoint_1['activation_fns']
        training_time = checkpoint_1['training_time']
    else:
        print("Warning: Aggregated model training info file not found.")

    # Load models
    for i in range(num_models):
        model = model_class(config_model)
        checkpoint_path = os.path.join(save_dir, f'model_{i}_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            loaded_models.append(model)
        else:
            print(f"Warning: Checkpoint for model {i} not found.")

    
    print(f"Loaded {len(loaded_models)} models.")
    return loaded_models, loaded_losses, hidden_sizes, activation_fns, training_time



# --------------------------------------------------------------------------- Method for early stopping 
# stopping criteria: use the quotient of generalization loss and progress.
def stop_training(val_losses, train_losses, patience, alpha):
    """
    Implements early stopping using: PQα
    Stop after first end-of-strip epoch t with GL(t)/Pk(t) > α
    
    Args:
        val_losses (list): History of validation losses
        train_losses (list): History of training losses
        alpha (float): Threshold for stopping criterion
        strip_length (int): Length of training strips
        
    Returns:
        bool: True if training should stop
    """


    if len(val_losses) < patience:  # Need enough points to calculate
        return False
        
    # Calculate generalization loss GL(t)
    best_loss = min(val_losses)
    current_loss = val_losses[-1]
    gl = 100 * ((current_loss / best_loss) - 1)
    
    # Calculate training progress Pk(t)
    recent_losses = train_losses[-patience:]
    strip_min = min(recent_losses)
    if strip_min == 0:
        return gl > 0
        
    strip_avg = sum(recent_losses) / patience
    pk = 1000 * ((strip_avg / strip_min) - 1)
    
    # Stop if GL(t)/Pk(t) > α
    return (gl / pk) > alpha if pk > 0 else gl > 0



# --------------------------------------------------------------------------- Methods to obtain aggregated predictions (mean) and uncertainty 
# Function to evaluate NN model
def get_average_predictions(networks, inputs_norm):
  n_output_features = 17
  num_networks = len(networks)

  # Test ensemble model - predictions are the average of the weak models' predictions
  avg_predictions = torch.zeros((inputs_norm.shape[0], n_output_features))
  inputs_norm_ = torch.tensor(inputs_norm, dtype=torch.float64)

  for model in networks:
    # 6. Evaluate: set mode
    model.eval()
    model.to(torch.double) 
    with torch.no_grad():
      predictions = model(inputs_norm_)
    avg_predictions += predictions

  avg_predictions /= num_networks

  return avg_predictions

# Function to get std/sqrt(N) in the aggregated models predictions
def get_predictive_uncertainty(networks, inputs_norm):
    num_networks = len(networks)

    # Test ensemble model - calculate the standard deviation of the weak models' predictions
    predictions_list = []

    for idx, model in enumerate(networks):
      # Evaluate: set model to evaluation mode
      model.eval()
      with torch.no_grad():
        predictions = model(inputs_norm)
      if torch.isnan(predictions).any():
        raise ValueError(f"NaNs found in predictions from model {idx}")
      predictions_list.append(predictions)
    
    # Stack predictions and check for NaNs
    stacked_predictions = torch.stack(predictions_list, dim=0)
    if torch.isnan(stacked_predictions).any():
        raise ValueError("NaNs found in stacked_predictions")

    # Calculate the standard deviation of the predictions
    std_dev_predictions = torch.std(stacked_predictions, dim=0, unbiased=False)  # Use unbiased=False to match NumPy's behavior
    if torch.isnan(std_dev_predictions).any():
        raise ValueError("NaNs found in std_dev_predictions")

    # Normalize the standard deviation by the square root of the number of networks
    std_dev_predictions /= torch.sqrt(torch.tensor(num_networks, dtype=torch.float64))
    if torch.isnan(std_dev_predictions).any():
        raise ValueError("NaNs found in normalized std_dev_predictions")

    error_predictions = std_dev_predictions

    return error_predictions

