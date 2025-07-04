import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.spring_mass_system.utils import set_seed, save_checkpoint
from src.spring_mass_system.utils import compute_total_energy, savefig

#set_seed(42)

activation_functions = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
}


class NeuralNetwork(nn.Module):
    def __init__(self, config_model):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        # Extract values from config file
        input_size = config_model['input_size']
        hidden_sizes = config_model['hidden_sizes']
        output_size = config_model['output_size']
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # get activation function 
        activation_fn = activation_functions[config_model['activation_fn']]
        
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


def train_model(config_, config_model, model, preprocessed_data, optimizer, device, checkpoint_dir, seed = 42, print_every=5, print_messages=True):
    model.to(device)
    train_losses, train_physics_losses, train_data_losses, val_losses, val_physics_losses = [], [], [], [], []
    best_val_loss = float('inf')

    if config_model['loss_physics_weight'] == 0:
        model_name = "NN"
    else:
        model_name = "PINN"

    # Access the variables from the config file
    num_epochs = config_model['num_epochs']

    # Access the components from the preprocessed data
    train_loader = preprocessed_data['train_loader']
    val_loader = preprocessed_data['val_loader']

    # Acess the scaler components from the preprocessed data
    scaler_X = preprocessed_data['scaler_X']

    # Ensure the directory exists
    if checkpoint_dir is not None:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    if print_messages:
        print(f"\nInitiating training of the {model_name} model ...")

    # Run epochs loop
    for epoch in tqdm(range(num_epochs), desc=f"Training {model_name}"):
        set_seed(seed)
        model.train()
        train_loss_dict = _run_epoch_train(config_, config_model, model, train_loader, optimizer, device, scaler_X, seed)

        # append values from dict
        train_losses.append(train_loss_dict['train_loss_total'])
        train_physics_losses.append(train_loss_dict['train_loss_physics'])
        train_data_losses.append(train_loss_dict['train_loss_data'])

        model.eval()
        with torch.no_grad():
            val_loss, val_physics = _run_epoch_val(model, val_loader, device, config_, config_model, scaler_X) 
            val_losses.append(val_loss)
            val_physics_losses.append(val_physics)

        if (config_['plotting']['PRINT_LOSS_VALUES'] == True):
            if epoch % print_every == 0:
                _print_epoch_summary(epoch, train_loss_dict, val_loss, val_physics)

        # Save checkpoint
        if checkpoint_dir is not None:
            save_checkpoint(
                model, optimizer, epoch, train_losses, train_data_losses, train_physics_losses, val_losses, val_physics_losses,
                os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            )
        
        # Save the best model based on validation loss
        if checkpoint_dir is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, train_losses, train_data_losses, train_physics_losses, val_losses, val_physics_losses,
                    os.path.join(checkpoint_dir, 'best_checkpoint.pth')
                )
    
    if print_messages:
        print(f"All checkpoints and aggregated losses saved in {checkpoint_dir}")

    if print_messages:
        print("Model training complete.\n\n")

    return {
        'train_losses': train_losses,
        'train_physics_losses': train_physics_losses,
        'train_data_losses': train_data_losses,
        'val_data_losses': val_losses,
        'val_physics_losses': val_physics_losses
    }


def _compute_pinn_loss(config_, config_model, inputs_norm, predictions_norm, target_norm, scaler_X):

    # revert normalization
    inputs = torch.tensor(scaler_X.inverse_transform(inputs_norm.cpu().numpy()), device=inputs_norm.device)
    predictions = torch.tensor(scaler_X.inverse_transform(predictions_norm.detach().cpu().numpy()), device=predictions_norm.device)
    
    # Compute energy
    energy_inputs = torch.tensor([compute_total_energy(config_, state) for state in inputs.numpy()])
    energy_outputs = torch.tensor([compute_total_energy(config_, state) for state in predictions.numpy()])

    # lambda_physics
    loss_physics_weight = config_model['loss_physics_weight']

    # compute loss physics
    loss_physics = torch.mean((energy_inputs - energy_outputs) ** 2)
    loss_physics_weighted = loss_physics_weight * loss_physics

    # compute loss data
    loss_data = nn.MSELoss()(predictions_norm, target_norm)
    loss_data_weighted = (1 - loss_physics_weight) * loss_data

    # compute weighted loss 
    loss_total_pinn = (1 - loss_physics_weight) * loss_data + loss_physics_weight * loss_physics

    return {
        'loss_physics': loss_physics,
        'loss_physics_weighted': loss_physics_weighted, 
        'loss_data': loss_data,
        'loss_data_weighted': loss_data_weighted, 
        'loss_total': loss_total_pinn
    }


def _run_epoch_train(config_, config_model, model, data_loader, optimizer, device, scaler_X, seed):
    loss_total_pinn = 0.0
    loss_physics = 0.0
    loss_data = 0.0
    loss_physics_weighted = 0.0
    loss_data_weighted = 0.0
    
    # for each epoch, rain the training loop
    for inputs, targets in data_loader:
        
        # Set seed 
        set_seed(seed)

        # get intputs and targets from loader
        inputs, targets = inputs.to(device), targets.to(device)

        # compute predicted outputs
        outputs = model(inputs)

        # compute loss
        loss_dict = _compute_pinn_loss(config_, config_model, inputs, outputs, targets, scaler_X)        
        
        # train
        if optimizer is not None:  # Training mode
            optimizer.zero_grad()
            loss_dict['loss_total'].backward()
            optimizer.step()
        
        # append loss results
        batch_size = inputs.size(0)
        loss_total_pinn += loss_dict['loss_total'].item() * batch_size
        loss_physics += loss_dict['loss_physics'].item() * batch_size
        loss_data += loss_dict['loss_data'].item() * batch_size
        loss_physics_weighted += loss_dict['loss_physics_weighted'].item() * batch_size
        loss_data_weighted += loss_dict['loss_data_weighted'].item() * batch_size

    num_samples = len(data_loader.dataset)
    
    return {
        'train_loss_total': loss_total_pinn / num_samples, 
        'train_loss_data': loss_data / num_samples, 
        'train_loss_physics': loss_physics / num_samples, 
        'train_weighted_physics_loss': loss_physics_weighted / num_samples,
        'train_weighted_data_loss':loss_data_weighted / num_samples
    }


def _run_epoch_val(model, data_loader, device, config_, config_model, scaler_X):
    val_total, val_physics_total = 0, 0

    for inputs, targets in data_loader:
        # get inputs and targets from data_loader
        inputs, targets = inputs.to(device), targets.to(device)

        # make predictions
        outputs = model(inputs)

        # compute validation loss
        loss_val = nn.MSELoss()(outputs, targets)
        loss_dict = _compute_pinn_loss(config_, config_model, inputs, outputs, targets, scaler_X)     
        
        # 
        batch_size = inputs.size(0)
        val_total += loss_val.item() * batch_size
        val_physics_total += loss_dict['loss_physics'].item() * batch_size
    
    val_total         = val_total / len(data_loader.dataset)
    val_physics_total = val_physics_total / len(data_loader.dataset)

    return val_total, val_physics_total


def _print_epoch_summary(epoch, train_loss_dict, val_loss, val_physics):
    print(f'Epoch {epoch}:')
    print(f'  Train Loss Total: {train_loss_dict["train_loss_total"]:.4e}')
    print(f'  Train Loss Physics ({_percentage(train_loss_dict["train_weighted_physics_loss"], train_loss_dict["train_loss_total"]):.2f}%): {train_loss_dict["train_loss_physics"]:.4e}')
    print(f'  Train Loss Data ({_percentage(train_loss_dict["train_weighted_data_loss"], train_loss_dict["train_loss_total"]):.2f}%): {train_loss_dict["train_loss_data"]:.4e}')
    print(f'  Val Loss Data: {val_loss:.4e}')
    print(f'  Val Loss Physics: {val_physics:.4e}')


def _percentage(part, whole):
    return (part / whole) * 100


def plot_loss_curves_(config, losses_dict, model_name):

    # extract losses lists from dict
    train_losses_total = losses_dict['train_losses']
    train_losses_physics = losses_dict['train_physics_losses']
    train_losses_data= losses_dict['train_data_losses']
    val_losses = losses_dict['val_data_losses']

    plt.figure(figsize=(10, 6))
    # Append total
    plt.plot(train_losses_total, label=r'Train Loss Total ($\mathcal{L}_{\mathrm{train}}$)', linewidth=2, color='#1f77b4')
    
    # Plot validation loss
    plt.plot(val_losses, label=r'Validation Loss ($\mathcal{L}_{\mathrm{val}}$)', linewidth=2, color='#ff7f0e',  linestyle='--')
    
    # Plot data loss
    plt.plot(train_losses_data, label=r'$\mathcal{L}_{\mathrm{data}}$', linewidth=2, color='#2ca02c')
    
    # Plot physics loss
    plt.plot(train_losses_physics, label=r'Weighted $\mathcal{L}_{\mathrm{physics}}$', linewidth=2, color='#d62728')
    # define style
    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)
    plt.xlabel('Epochs', fontweight='bold', fontsize=16)
    plt.ylabel(r'$\mathcal{L}_{\mathrm{MAE}}$', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale('log')  
    plt.tight_layout()

    # Save figures
    output_dir = config['plotting']['output_dir'] + "loss_curves/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_name}_loss_curves")
    savefig(save_path, pad_inches = 0.2)
