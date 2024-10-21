import os
import re
import pdb
import time
import yaml
import numpy as np
from typing import Dict, Any, Tuple
import logging

from src.ltp_system.dataset_gen import simulations as simul_class

# Generate uniform input sample (P, I, R)     --------------------------------------------------------
def get_input_array(n_simulations, input_ref, bounds):

    pressure_set = input_ref[0] * np.random.uniform(bounds[0][0], bounds[0][1], size = n_simulations)
    ne_set = input_ref[1] * np.random.uniform(bounds[1][0], bounds[1][1], size = n_simulations)
    R_set = input_ref[2] * np.random.uniform(bounds[2][0], bounds[2][1], size = n_simulations)
    
    sample = np.stack((pressure_set, ne_set, R_set), axis=1)

    return sample

# Load the configuration file  -----------------------------------------------------------------------
def load_config(config_path):
    
    # Load the configuration file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Define the name of the output file with the dataset
def generate_filename(n_simulations, simulation_params):
    base = f'/data_{n_simulations}_points'
    
    if simulation_params['const_current_simul'] == 1:
        input_current = simulation_params['input_current']
        return f'/const_current{base}_{input_current}mA_NEW.txt'
    
    elif simulation_params['const_pressure_simul'] == 1:
        input_pressure = simulation_params['input_pressure']
        return f'/const_pressure{base}_{input_pressure}Torr_NEW.txt'
    
    elif simulation_params['const_current_simul'] == 0 and simulation_params['const_pressure_simul'] == 0:
        return f'{base}_NEW.txt'
    
    else:
        raise ValueError("Invalid simulation type")

def get_simulation_input():
    const_current_simul = input("Want a simulation with constant current? (y/n): ").lower() == 'y'
    const_pressure_simul = False
    input_current = input_pressure = 0
    I_range = p_range = R_range = []

    if const_current_simul:
        input_current = int(input("Insert the constant current in mA: "))
        I_range = [input_current, input_current]
        R_range = [1.2, 1.2]
    else:
        const_pressure_simul = input("Want a simulation with constant pressure? (y/n): ").lower() == 'y'
        if const_pressure_simul:
            input_pressure = int(input("Insert the constant pressure in Torr: "))
            p_range = [input_pressure, input_pressure]
            R_range = [1.2, 1.2]

    return {
        'const_current_simul': const_current_simul,
        'const_pressure_simul': const_pressure_simul,
        'input_current': input_current,
        'input_pressure': input_pressure,
        'I_range': I_range,
        'p_range': p_range,
        'R_range': R_range
    }

# Generate LoKI setup files - need to go to Matlab to run the LoKI code
def get_dataset(config, input_ref, p_range, I_range, R_range):

    # 1. extract data from config
    n_simulations = config['dataset_generation']['n_simulations']

    # 2. get user input regarding the bounds of the dataset to generate 
    try:
        simulation_params = get_simulation_input()
        print("Simulation parameters:", simulation_params)
    except ValueError as e:
        print(f"Error: Invalid input. {e}")

    if simulation_params['const_current_simul']:
        I_range = simulation_params['I_range']
        R_range = simulation_params['R_range']
    elif simulation_params['const_pressure_simul']:
        p_range = simulation_params['p_range']
        R_range = simulation_params['R_range']
    
    bounds = np.array([p_range, I_range, R_range]) 

    # 3. generate input samples
    inputs_arr = get_input_array(n_simulations, input_ref, bounds) 
    print("Pressure bounds: ", np.format_float_scientific(np.min(np.array(inputs_arr[:,0]))), " to ", np.format_float_scientific(np.max(np.array(inputs_arr[:,0]))))
    print("I bounds      : ", np.format_float_scientific(np.min(np.array(inputs_arr[:,1]))), " to ", np.format_float_scientific(np.max(np.array(inputs_arr[:,1]))))
    print("R bounds       : ", np.format_float_scientific(np.min(np.array(inputs_arr[:,2]))), " to ", np.format_float_scientific(np.max(np.array(inputs_arr[:,2]))))

    if simulation_params['const_current_simul'] == 1:
        inputs_arr[:,0] = np.linspace(p_range[0]*input_ref[0], p_range[1]*input_ref[0], n_simulations)
    if simulation_params['const_pressure_simul'] == 1:
        inputs_arr[:,1] = np.linspace(I_range[0]*input_ref[1], I_range[1]*input_ref[1], n_simulations)

    p_torr = inputs_arr[:,0] * 0.00750062 # conversion Pa -> Torr
    I_mA = inputs_arr[:,1] * 1000         # conversion A  -> mA
    R_mm = inputs_arr[:,2] * 100          # conversion m  -> cm

    # expression for the initial condition of the electron density
    ne_column = ((2.189e+14 * p_torr**1.5 - 1.693e+15 * p_torr + 4.877e+15 * p_torr**0.5 + 1.158e+15) * I_mA / 30 * 1 / R_mm**1.5).reshape(-1, 1)
    inputs_arr = np.hstack((inputs_arr, ne_column))

    # 4. create the simulations object needed to create the setup files for LoKI
    simul = simul_class.Simulations(n_simulations, inputs_arr, config)

    # define the name of the dataset that is being generated
    try:
        filename = generate_filename(n_simulations, simulation_params)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)

    # 5. turn off/on for fixed/changing values of k's
    simul.set_ChemFile_OFF() 

    # 6. call LoKI and Generate Outputs
    simul.runSimulations()
    simul.writeDataFile(filename)


#-----------------------------------------------------------------------------------------------------
def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

        # get config
        config = load_config('configs/ltp_system_config.yaml')
        
        input_ref = [133.322, 0.001, 0.01]
        p_range  = [0.1, 10]        # [1.3332e1, 1.3332e3] Pa
        I_range = [5, 50]           # [2       , 44]     mA
        R_range  = [0.4,  2]        # [4e-3    , 2e-2]     m

        get_dataset(config, input_ref, p_range, I_range, R_range)
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()