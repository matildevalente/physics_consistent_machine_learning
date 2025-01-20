import os
import yaml
import torch
import yaml
import numpy as np
import matplotlib as mpl
import torch.nn as nn
from tqdm import tqdm
from skopt.space import Integer, Categorical
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.utils import use_named_args

from src.spring_mass_system.utils import set_seed, save_checkpoint
from src.spring_mass_system.utils import compute_total_energy, savefig, compute_parameters


activation_functions = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
}


# Define the NN model class
class PhysicsInformedNN(nn.Module):
    def __init__(self, config):
        super(PhysicsInformedNN, self).__init__()
        self.layers = nn.ModuleList()

        # Extract values from config file
        input_size = config['pinn_model']['input_size']
        hidden_sizes = config['pinn_model']['hidden_sizes']
        output_size = config['pinn_model']['output_size']
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # get activation function 
        activation_fn = activation_functions[config['pinn_model']['activation_fn']]

        
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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Train NN model
def train_pinn(config, model, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir, print_every=5):
    model.to(device)
    train_losses, train_physics_losses, train_data_losses, val_losses = [], [], [], []
    best_val_loss = float('inf')

    # Access the variables from the config file
    num_epochs = config['pinn_model']['num_epochs']

    # Access the components from the preprocessed data
    train_loader = preprocessed_data['train_loader']
    val_loader = preprocessed_data['val_loader']

    # Acess the scaler components from the preprocessed data
    scaler_X = preprocessed_data['scaler_X']

    # Ensure the directory exists
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    # Run epochs loop
    for epoch in tqdm(range(num_epochs), desc="Training PINN"):
        set_seed(42)
        model.train()
        train_loss_dict = _run_epoch_train(config, model, train_loader, loss_fn, optimizer, device, scaler_X)

        # append values from dict
        train_losses.append(train_loss_dict['train_loss'])
        train_physics_losses.append(train_loss_dict['train_weighted_physics_loss'])
        train_data_losses.append(train_loss_dict['train_weighted_data_loss'])

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch_val(model, val_loader, loss_fn, None, device)
            val_losses.append(val_loss)

        if (config['plotting']['PRINT_LOSS_VALUES'] == True):
            if epoch % print_every == 0:
                _print_epoch_summary(epoch, train_loss_dict, val_loss)

        # Save checkpoint
        if checkpoint_dir is not None:
            save_checkpoint(
                model, optimizer, epoch, train_losses, train_data_losses, train_physics_losses, val_losses, 
                os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            )
        
        # Save the best model based on validation loss
        if checkpoint_dir is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_losses, train_data_losses, train_physics_losses, val_losses, 
                    os.path.join(checkpoint_dir, 'best_checkpoint.pth')
                )
    
    return {
        'train_losses': train_losses,
        'train_physics_losses': train_physics_losses,
        'train_data_losses': train_data_losses,
        'val_losses': val_losses
    }


def _compute_pinn_loss(config, inputs_norm, predictions_norm, target_norm, scaler_X, loss_fn):

    # revert normalization
    inputs = torch.tensor(scaler_X.inverse_transform(inputs_norm.cpu().numpy()), device=inputs_norm.device)
    predictions = torch.tensor(scaler_X.inverse_transform(predictions_norm.detach().cpu().numpy()), device=predictions_norm.device)
    
    # Compute energy
    energy_inputs = torch.tensor([compute_total_energy(config, state) for state in inputs.numpy()])
    energy_outputs = torch.tensor([compute_total_energy(config, state) for state in predictions.numpy()])

    # compute loss physics
    loss_physics = torch.mean((energy_inputs - energy_outputs) ** 2)

    # compute loss data
    loss_data = loss_fn(predictions_norm, target_norm)

    # compute weighted loss 
    loss_physics_weight = config['pinn_model']['loss_physics_weight']
    loss_total_pinn = (1 - loss_physics_weight) * loss_data + loss_physics_weight * loss_physics

    return {
        'loss_physics': loss_physics,
        'loss_data': loss_data,
        'loss_total_pinn': loss_total_pinn
    }


def _run_epoch_train(config, model, data_loader, loss_fn, optimizer, device, scaler_X):
    loss_total_pinn = 0.0
    loss_physics = 0.0
    loss_data = 0.0
    loss_physics_weight = config['pinn_model']['loss_physics_weight']
    
    # for each epoch, rain the training loop
    for inputs, targets in data_loader:
        # Set seed 
        set_seed(42)

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss_dict = _compute_pinn_loss(config, inputs, outputs, targets, scaler_X, loss_fn)        
        
        if optimizer is not None:  # Training mode
            optimizer.zero_grad()
            loss_dict['loss_total_pinn'].backward()
            optimizer.step()
        
        batch_size = inputs.size(0)
        loss_total_pinn += loss_dict['loss_total_pinn'].item() * batch_size
        loss_physics += loss_physics_weight * loss_dict['loss_physics'].item() * batch_size
        loss_data += (1 - loss_physics_weight) * loss_dict['loss_data'].item() * batch_size

    num_samples = len(data_loader.dataset)
    
    return {
        'train_loss': loss_total_pinn / num_samples, 
        'train_weighted_physics_loss': loss_physics / num_samples, 
        'train_weighted_data_loss': loss_data / num_samples
    }


def _run_epoch_val(model, data_loader, loss_fn, optimizer, device):
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(data_loader.dataset)


def _print_epoch_summary(epoch, train_loss_dict, val_loss):
    print(f'Epoch {epoch}:')
    print(f'  Train Loss Total: {train_loss_dict["train_loss"]:.4e}')
    print(f'  Train Loss Physics ({_percentage(train_loss_dict["train_weighted_physics_loss"], train_loss_dict["train_loss"]):.2f}%): {train_loss_dict["train_weighted_physics_loss"]:.4e}')
    print(f'  Train Loss Data ({_percentage(train_loss_dict["train_weighted_data_loss"], train_loss_dict["train_loss"]):.2f}%): {train_loss_dict["train_weighted_data_loss"]:.4e}')
    print(f'  Val Loss: {val_loss:.4e}')


def _percentage(part, whole):
    return (part / whole) * 100


# This function runs if the user defines PLOT_LOSS_CURVES: True it in the config file
def plot_loss_curves_pinn(config, pinn_losses_dict):

    # extract losses lists from dict
    train_losses_total = pinn_losses_dict['train_losses']
    train_losses_physics = pinn_losses_dict['train_physics_losses']
    train_losses_data= pinn_losses_dict['train_data_losses']
    val_losses = pinn_losses_dict['val_losses']

    plt.figure(figsize=(10, 6))
    # Append total
    plt.plot(train_losses_total, label=r'Train Loss Total ($\mathcal{L}_{\mathrm{train}}$)', linewidth=2, color='#1f77b4')
    
    # Plot validation loss
    plt.plot(val_losses, label=r'Validation Loss ($\mathcal{L}_{\mathrm{val}}$)', linewidth=2, color='#ff7f0e',  linestyle='--')
    
    # Plot data loss
    plt.plot(train_losses_data, label=r'Weighted $\mathcal{L}_{\mathrm{data}}$', linewidth=2, color='#2ca02c')
    
    # Plot physics loss
    plt.plot(train_losses_physics, label=r'Weighted $\mathcal{L}_{\mathrm{physics}}$', linewidth=2, color='#d62728')
    # define style
    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)
    plt.xlabel('Epochs', fontweight='bold', fontsize=16)
    plt.ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale('log')  
    plt.tight_layout()

    # Save figures
    output_dir = config['plotting']['output_dir'] + "loss_curves/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"pinn_loss_curves")
    savefig(save_path, pad_inches = 0.2)


"""### This section of the code optimizes the PINN architecture based on the val loss
def generate_config_(config, hidden_sizes, activation_fn):
    return {
        'pinn_model': {
            'RETRAIN_MODEL': config['pinn_model']['RETRAIN_MODEL'],
            'input_size': config['pinn_model']['input_size'],
            'hidden_sizes': hidden_sizes,
            'output_size': config['pinn_model']['output_size'],
            'num_epochs': config['pinn_model']['num_epochs'],
            'learning_rate': config['pinn_model']['learning_rate'],
            'activation_fn': activation_fn,
            'loss_physics_weight': config['pinn_model']['loss_physics_weight'],
        },
        'spring_mass_system': config['spring_mass_system'],
        'plotting': config['plotting']
    }
    
def evaluate_architecture(config, architecture, preprocessed_data, activation_function):

    # create pinn model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = config['pinn_model']['learning_rate']
    config_new = generate_config_(config, architecture, activation_function)
    pinn_model = PhysicsInformedNN(config_new).to(device)
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=learning_rate)

    # train pinn model
    losses_dict = train_pinn(config_new, pinn_model, preprocessed_data, nn.MSELoss(), optimizer, device, None)
    
    return losses_dict['val_losses'][-1]

def optimize_pinn_architecture(original_config, preprocessed_data):
    # Define the search space
    search_space  = [
        Integer(1, 150, name='layer_1'), 
        Integer(1, 150, name='layer_2'), 
        Integer(1, 150, name='layer_3'),   
        Integer(1, 150, name='layer_4'),
        Categorical(['relu', 'tanh', 'sigmoid', 'leaky_relu'], name='activation')
    ]

    # Counter to track the iterations
    iteration = 0
    val_loss_list, iteration_list, architecture_list, activ_func_list = [], [], [], []

    # Use @use_named_args to match the function signature with the space definition
    @use_named_args(search_space)
    def objective(**params):
        nonlocal iteration
        iteration += 1
        architecture = [params['layer_1'], params['layer_2'], params['layer_3'], params['layer_4']]
        activation_function = params['activation']
        # Evaluate the architecture
        val_loss = evaluate_architecture(original_config, architecture, preprocessed_data, activation_function)
        #print(f"Iteration {iteration}: Architecture: {architecture}, Activation: {activation_function}, Validation Loss: {val_loss}")
        val_loss_list.append(val_loss)
        iteration_list.append(iteration)

        return val_loss
    
    # Now run Bayesian optimization
    result = gp_minimize(objective, search_space, n_calls=35, random_state=42 )

    best_architecture = result.x[:4]  
    best_activation = result.x[4]     

    plot_architecture_optimization(original_config, iteration_list, val_loss_list, best_architecture, best_activation)
    
    opt_config = generate_config_(original_config, best_architecture, best_activation)
    
    save_config_(opt_config)

    pinn_model_opt, losses = get_optimized_pinn(opt_config, preprocessed_data)
    
    return pinn_model_opt, losses, opt_config

def get_optimized_pinn(opt_config, preprocessed_data):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    pinn_model_opt = PhysicsInformedNN(opt_config).to(device)
    loss_fn = nn.MSELoss()
    learning_rate = opt_config['pinn_model']['learning_rate']
    optimizer = torch.optim.Adam(pinn_model_opt.parameters(), lr=learning_rate)
    
    checkpoint_dir = os.path.join('output', 'spring_mass_system', 'checkpoints', 'pinn_opt')
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    losses = train_pinn(opt_config, pinn_model_opt, preprocessed_data, loss_fn, optimizer, device, checkpoint_dir)

    return pinn_model_opt, losses

def plot_architecture_optimization(config, iteration_list, val_loss_list, best_architecture, best_activation):

    num_parameters_nn = compute_parameters(config['nn_model']['hidden_sizes'])
    num_parameters_pinn = compute_parameters(best_architecture)
    compare_rate = 100 * ( num_parameters_pinn - num_parameters_nn ) / num_parameters_nn

    # Plot validation loss as a function of iterations
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_list, val_loss_list, marker='o', linestyle='-', label='Validation Loss')
    best_iteration = iteration_list[val_loss_list.index(min(val_loss_list))]
    plt.scatter(best_iteration, min(val_loss_list), color='green', s=100, zorder=5, label='Best Loss')
    plt.yscale('log')
    plt.ylim([min(val_loss_list) * 0.8, max(val_loss_list) * 10])
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Validation Loss', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"No Parameters in the PINN compared to NN: {compare_rate:.2f} $\%$ ({best_activation} , {best_architecture}) ", fontsize=9, color='gray')
    plt.legend(fontsize=14)
    # Save figures
    output_dir = config['plotting']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "extra_figures/Extra_Figure_2")
    savefig(save_path, pad_inches = 0.2)

def save_config_(config, filename='pinn_opt_config.yaml'):

    def numpy_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: numpy_to_python(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        return obj
    
    def represent_list(dumper, data):
        # Check if all items in the list are scalars (int, float, str)
        if all(isinstance(item, (int, float, str)) for item in data):
            # Represent as a flow-style sequence
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        # For nested lists or other complex types, use the default representation
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)


    # Ensure the directory exists
    os.makedirs("configs/", exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join("configs/", filename)
    
    # Convert numpy types to Python types
    python_config = numpy_to_python(config)
    
    # Create a new YAML dumper
    class CompactDumper(yaml.SafeDumper):
        pass
    
    # Add the custom representer to our dumper
    CompactDumper.add_representer(list, represent_list)
    
    # Write the config to the file
    with open(file_path, 'w') as f:
        yaml.dump(python_config, f, Dumper=CompactDumper, default_flow_style=False)
    
"""