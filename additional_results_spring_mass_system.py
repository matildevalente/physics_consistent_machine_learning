import os
import yaml
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.spring_mass_system.plotter.Figure_3 import plot_several_initial_conditions


# Load the configuration file
def load_config(config_path):
    
    # Load the configuration file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_rmse_comparison(config, solver_tolerances, rmse_pinn, std_pinn, rmse_nn, std_nn, 
                        figsize=(8, 6), dpi=300):
    """
    Create a dual-axis plot comparing RMSE values between PINN and NN projections.
    
    Parameters:
    -----------
    solver_tolerances : array-like
        Array of solver tolerance values
    rmse_pinn : array-like
        RMSE values for PINN projection
    std_pinn : array-like
        Standard deviation values for PINN projection
    rmse_nn : array-like
        RMSE values for NN projection
    std_nn : array-like
        Standard deviation values for NN projection
    save_path : str, optional
        Path where to save the plot (default: 'error_comparison_plot.pdf')
    figsize : tuple, optional
        Figure size in inches (default: (8, 6))
    dpi : int, optional
        Resolution of the figure (default: 300)
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    (ax1, ax2) : tuple
        The primary and secondary axis objects
    """
    # Calculate differences
    diffs = np.abs(rmse_nn - rmse_pinn)
    
    # Set figure DPI
    plt.rcParams['figure.dpi'] = dpi
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot on primary y-axis
    ax1.errorbar(solver_tolerances, rmse_pinn, yerr=std_pinn, fmt='o', color='purple', ecolor='black', capsize=5, capthick=1, markersize=8, label='PINN Projection')
    ax1.errorbar(solver_tolerances, rmse_nn, yerr=std_nn, fmt='x', color='red', ecolor='black', capsize=5, capthick=1, markersize=10, markeredgewidth=2,label='NN Projection')
    # Set scales
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_ylim(1e-6, 5e-5)  # for example, from 10^-6 to 10^-5


    # Set primary y-axis labels and properties
    ax1.set_xlabel('Tolerance of Projection Solver', fontsize=20)
    ax1.set_ylabel('Mean RMSE in Energy Prediction', color='black', fontsize=20)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=20)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=20)
    
    # Set x-axis ticks to match solver_tolerances
    ax1.set_xticks(solver_tolerances)
    ax1.set_xticklabels([f'1E{int(np.log10(x))}' for x in solver_tolerances])

    # Create second y-axis and plot differences
    ax2 = ax1.twinx()
    ax2.plot(solver_tolerances, diffs, 'g--', linewidth=2, 
             label=r'$|\mathrm{RMSE_{NN}} - \mathrm{RMSE_{PINN}}|$')
    ax2.scatter(solver_tolerances, diffs, color='green', marker='s')
    
    # Set secondary y-axis labels and properties
    ax2.set_ylabel(r'$|\mathrm{RMSE_{NN}} - \mathrm{RMSE_{PINN}}|$', color='green', fontsize=20)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=20)
    
    # Add title
    plt.title('Mean RMSE of Energy vs Solver Tolerance', fontsize=20)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=20)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-13, 1e-4)

    # Adjust layout
    plt.tight_layout()
    
    # Construct the full save path
    save_dir = os.path.join(config['plotting']['output_dir'], 'additional_results')
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    save_path = os.path.join(save_dir, 'Energy_rmse_vs_solver_tolerance.pdf')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig, (ax1, ax2)


def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # Load the configuration file
        config = load_config('configs/spring_mass_system_config.yaml')

        # Tolerances of the projection solver
        solver_tolerances = np.array([1E-9, 1E-8, 1E-7, 1E-6, 1E-3])

        #plot_several_initial_conditions(config, preprocessed_data, nn_model, pinn_model, test_initial_conditions, n_time_steps, N_initial_conditions = 100)

        # Mean RMSE across 100 trajectories in energy prediction
        rmse_pinn = np.array([7.98251438E-06, 7.98252087E-06, 7.98259411E-06, 7.98502790E-06, 2.00876819E-05])
        rmse_nn   = np.array([7.98251378E-06, 7.98251820E-06, 7.98267183E-06, 7.98795132E-06, 1.96008239E-05])

        # Standard Deviation across 100 trajectories in energy prediction
        std_pinn  = np.array([6.12558665e-06, 6.12559535E-06, 6.12558077E-06, 6.12609402E-06, 7.16595988E-06])  
        std_nn    = np.array([6.12558570E-06, 6.12558964E-06, 6.12560497E-06, 6.12553577E-06, 6.84902939E-06])  
        
        # Create the plot
        plot_rmse_comparison(config, solver_tolerances, rmse_pinn, std_pinn, rmse_nn, std_nn)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise



if __name__ == "__main__":
    main()
    