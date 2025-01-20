import numpy as np
import pandas as pd
from tqdm import tqdm

from src.spring_mass_system.utils import compute_total_energy, rk4_step, set_seed


# Function to generate random initial conditions with energy check
def random_initial_conditions(config, E_MAX):
    
    while True:
        x1_0 = np.random.uniform(-5, 5)
        x2_0 = np.random.uniform(-5, 5)
        v1_0 = np.random.uniform(-5, 5)
        v2_0 = np.random.uniform(-5, 5)
        
        energy_in = compute_total_energy(config, [x1_0, v1_0, x2_0, v2_0])
        
        if energy_in < E_MAX:
            return [x1_0, v1_0, x2_0, v2_0]


# Generate dataset between Emin and Emax and return dataframe
def generate_dataset(config, column_names):
              
    # Loop to generate dataset
    data, energy_list = [], []

    # Extract variables for dataset generation from the config
    dt_RK = config['dataset_generation']['dt_RK']
    E_MAX = config['dataset_generation']['E_MAX']
    num_samples = config['dataset_generation']['num_samples']
    N_sequential_steps = config['dataset_generation']['N_RK_STEPS']

    # Set seed 
    set_seed(42)
    
    # Loop to generate dataset with tqdm progress bar
    for _ in tqdm(range(num_samples), desc="Generating Dataset"):
        
        # Select an energy bin with the minimum count
        initial_state = random_initial_conditions(config, E_MAX)

        # Make the initial state the current_state before entering the cycle
        current_state = initial_state
        
        # Enter the cycle to compute the next state
        for _ in range(N_sequential_steps):
            
            # Solve from t_start to t_start + dt
            next_state = rk4_step(config, current_state, dt_RK)

            # Update next state
            current_state = next_state
        
        # Compute energy error
        energy_out = compute_total_energy(config, [next_state[0], next_state[1], next_state[2], next_state[3]])
        
        # Combine initial conditions and state after dt_RK
        combined = np.hstack([initial_state, next_state])
        data.append(combined)
        energy_list.append(energy_out)

    
    # Create dataframe with dataset 
    df = pd.DataFrame(data, columns = column_names)

    # Add an extra column for the total energy
    df['E'] = energy_list

    # Save to local directory
    df.to_csv("data/spring_mass_system/data.csv", index=False)
    
    return df

