import os
import torch
import yaml
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib as mpl
from skopt.space import Integer, Categorical
from skopt import gp_minimize
from skopt.utils import use_named_args
from typing import Optional, Tuple, List
from src.spring_mass_system.utils import set_seed, save_checkpoint, savefig

set_seed(42)

activation_functions = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU()
}


# Define the NN model class
class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Extract values from config file
        input_size = config['nn_model']['input_size']
        hidden_sizes = config['nn_model']['hidden_sizes']
        output_size = config['nn_model']['output_size']
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # get activation function 
        activation_fn = activation_functions[config['nn_model']['activation_fn']]
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(activation_fn)
        
        self.init_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)


# Train NN model
def train_nn(config, model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir, print_every=5):
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    num_epochs = config['nn_model']['num_epochs']
    train_loader = preprocessed_data['train_loader']
    val_loader = preprocessed_data['val_loader']

    # Ensure the directory exists
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


    for epoch in tqdm(range(num_epochs), desc="Training NN  "):
        set_seed(42)
        model.train()
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, loss_fn, None, device)
            val_losses.append(val_loss)
        
        if (config['plotting']['PRINT_LOSS_VALUES'] == True):
            if epoch % print_every == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}')

        # Save checkpoint
        if checkpoint_dir is not None:
            save_checkpoint(
                model, optimizer, epoch, train_losses, [], [], val_losses, 
                os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            )

        # Save the best model based on validation loss
        if checkpoint_dir is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_losses, [], [], val_losses, 
                    os.path.join(checkpoint_dir, 'best_checkpoint.pth')
                )

    return train_losses, val_losses

# Run an epoch
def _run_epoch(model, data_loader, loss_fn, optimizer, device):
    total_loss = 0
    for inputs, targets in data_loader:
        # Set seed 
        set_seed(42)
        
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        if optimizer is not None:  # Training mode
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(data_loader.dataset)

# This function runs if the user defines PLOT_LOSS_CURVES: True it in the config file
def plot_loss_curves_nn(config, train_losses, val_losses):

    # Plot in log scale
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label=r'Train Loss ($\mathcal{L_{\text{train}}}$)', linewidth=2, color='#1f77b4')
    plt.plot(val_losses, label=r'Validation Loss ($\mathcal{L_{\text{val}}}$)', linewidth=2,  linestyle='--', color='#ff7f0e')
    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)
    plt.xlabel('Epochs', fontweight='bold', fontsize=16)
    plt.ylabel(r'$\mathcal{L_{\text{MSE}}}$', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.tight_layout()

    # Save figures
    output_dir = config['plotting']['output_dir'] + "loss_curves/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"nn_loss_curves")
    savefig(save_path, pad_inches = 0.2)




### This section of the code optimizes the PINN architecture based on the val loss
def generate_config_(config, hidden_sizes, activation_fn):
    return {
        'nn_model': {
            'RETRAIN_MODEL': config['nn_model']['RETRAIN_MODEL'],
            'input_size': config['nn_model']['input_size'],
            'hidden_sizes': hidden_sizes,
            'output_size': config['nn_model']['output_size'],
            'num_epochs': config['nn_model']['num_epochs'],
            'learning_rate': config['nn_model']['learning_rate'],
            'activation_fn': activation_fn,
        },
        'spring_mass_system': config['spring_mass_system'],
        'plotting': config['plotting']
    }
    
def evaluate_architecture(config, architecture, preprocessed_data, activation_function):

    # create nn model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = config['nn_model']['learning_rate']
    config_new = generate_config_(config, architecture, activation_function)
    nn_model = NeuralNetwork(config_new).to(device)
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)

    # train pinn model
    train_losses, val_losses = train_nn(config_new, nn_model, preprocessed_data, nn.MSELoss(), optimizer, device, None)
    
    return val_losses[-1]

def optimize_architecture_nn(original_config, preprocessed_data, num_iterations):
    # Define the search space
    search_space  = [
        Integer(1, 150, name='layer_1'), 
        Integer(1, 150, name='layer_2'), 
        Integer(1, 150, name='layer_3'),   
        Categorical(['relu', 'tanh',  'leaky_relu'], name='activation')
    ]

    # Counter to track the iterations
    iteration = 0
    rows = []
    df_ = pd.DataFrame(columns= ['iteration', 'architecture', 'activation', 'val_loss'])

    # Use @use_named_args to match the function signature with the space definition
    @use_named_args(search_space)
    def objective(**params):
        nonlocal iteration
        iteration += 1
        architecture = [params['layer_1'], params['layer_2'], params['layer_3']]
        activation_function = params['activation']
        # Evaluate the architecture
        val_loss = evaluate_architecture(original_config, architecture, preprocessed_data, activation_function)

        new_row = {
            'iteration': iteration,
            'architecture': architecture, 
            'activation': activation_function, 
            'val_loss': val_loss
        }
        rows.append(new_row)
        
        return val_loss
    
    # Now run Bayesian optimization
    result = gp_minimize(objective, search_space, n_calls=num_iterations, random_state=42 )
    df_ = pd.DataFrame(rows, columns=['iteration', 'architecture', 'activation', 'val_loss'])

    print("Best NN architecture: ", result.x[:3]) 
    print("Best NN activation  : ", result.x[3] ) 
    print(df_) 
        

