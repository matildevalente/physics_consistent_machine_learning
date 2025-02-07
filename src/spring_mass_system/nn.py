"""import os
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

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

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Train NN model
def train_nn(config, model, preprocessed_data, optimizer, device, checkpoint_dir, print_every=5, print_messages=True):
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    # Access the variables from the config file
    num_epochs = config['nn_model']['num_epochs']

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
        print(f"\nInitiating training of the NN model ...")

    for epoch in tqdm(range(num_epochs), desc="Training NN  "):
        set_seed(42)
        model.train()
        train_loss = _run_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, None, device)
            val_losses.append(val_loss)
        
        if (config['plotting']['PRINT_LOSS_VALUES'] == True):
            if epoch % print_every == 0:
                #print(f'Epoch {epoch}: Train Loss: {train_loss:.6e}, Val Loss: {val_loss:.6e}')
                print(f'Epoch {epoch}:')
                print(f'  Train Loss Total          : {train_loss:.4e}')
                print(f'  Train Loss Physics (0.00%): 0.00')
                print(f'  Train Loss Data (0.00%)   : 0.00')
                print(f'  Val Loss                  : {val_loss:.4e}')

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

    if print_messages:
        print(f"All checkpoints and aggregated losses saved in {checkpoint_dir}")

    if print_messages:
        print("Model training complete.\n\n")

        
    return train_losses, val_losses


# Run an epoch
def _run_epoch(model, data_loader, optimizer, device):
    total_loss = 0
    for inputs, targets in data_loader:
        # Set seed 
        set_seed(42)
        
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, targets)
        
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
    plt.plot(train_losses, label=r'Train Loss ($\mathcal{L}_{\mathrm{train}}$)', linewidth=2, color='#1f77b4')
    plt.plot(val_losses, label=r'Validation Loss ($\mathcal{L}_{\mathrm{val}}$)', linewidth=2,  linestyle='--', color='#ff7f0e')
    plt.xticks(fontweight='bold', fontsize=14)
    plt.yticks(fontweight='bold', fontsize=14)
    plt.xlabel('Epochs', fontweight='bold', fontsize=16)
    plt.ylabel(r'$\mathcal{L}_{\mathrm{MSE}}$', fontweight='bold', fontsize=16)
    plt.legend(fontsize=12)
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.tight_layout()

    # Save figures
    output_dir = config['plotting']['output_dir'] + "loss_curves/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"nn_loss_curves")
    savefig(save_path, pad_inches = 0.2)

"""