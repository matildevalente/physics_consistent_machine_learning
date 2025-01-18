import os
import yaml
import torch
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple
import matplotlib.pyplot as plt
from src.ltp_system.data_prep import LoadDataset


logger = logging.getLogger(__name__)


# Load the configuration file
def load_config(config_path):
    
    # Load the configuration file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


# Set seed to garantee reproduc.. of results
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Load the dataset from a .csv file or generate the entire dataset
def load_dataset(config, dataset_dir):

    try:
        # Extract Data From File
        full_dataset = LoadDataset(dataset_dir)

        df = pd.read_csv(
            dataset_dir, 
            delimiter= config['dataset_generation']['delimiter'], 
            header=None, 
            names= config['dataset_generation']['input_features'] + config['dataset_generation']['output_features']
        )
        
        return df, full_dataset
    except FileNotFoundError:
        raise FileNotFoundError("Dataset file not found. Please generate the dataset or provide the correct file path.")

# 
def select_random_rows(input_file, dataset_size, seed, sampled_dataset = 'data/ltp_system/temp.txt'):

    try:
        # Set the random seed to ensure different samples for different seeds
        random.seed(seed)
        
        # Read all lines from the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        # Ensure N is not larger than the number of lines
        dataset_size = min(dataset_size, len(lines))
        
        # Randomly select N lines using the seeded random state
        selected_lines = random.sample(lines, dataset_size)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(sampled_dataset), exist_ok=True)
        
        # Write the selected lines to the temp file
        with open(sampled_dataset, 'w') as f:
            f.writelines(selected_lines)
        
        # Reset the random seed to avoid affecting other code
        random.seed()
        
        return sampled_dataset
    except FileNotFoundError:
        return False, f"Error: The input file '{input_file}' was not found."
    except PermissionError:
        return False, f"Error: Permission denied when trying to create or write to '{sampled_dataset}'."
    except Exception as e:
        return False, f"An unexpected error occurred: {str(e)}"


#
def split_dataset(input_file, n_testing_points, output_dir=None, testing_file=None, training_file=None):
    try:
        # Read all lines from the input file
        with open(input_file, 'r') as f:
            lines = f.readlines()

        # Ensure n_testing_points is not larger than the total number of lines
        total_lines = len(lines)
        if n_testing_points >= total_lines:
            raise ValueError(f"n_testing_points ({n_testing_points}) must be less than the total number of lines in the dataset ({total_lines})")

        # Randomly select n_testing_points for the testing set
        testing_indices = set(random.sample(range(total_lines), n_testing_points))

        # Determine output file paths
        if output_dir is None:
            output_dir = os.path.dirname(input_file)
        os.makedirs(output_dir, exist_ok=True)

        if testing_file is None:
            testing_file = os.path.join(output_dir, 'testing_data.txt')
        if training_file is None:
            training_file = os.path.join(output_dir, 'training_data.txt')

        # Write the data to testing and training files
        with open(testing_file, 'w') as test_f, open(training_file, 'w') as train_f:
            for i, line in enumerate(lines):
                if i in testing_indices:
                    test_f.write(line)
                else:
                    train_f.write(line)

        #print(f"Dataset split successfully:")
        #print(f"- Testing data ({n_testing_points} points) saved to: {testing_file}")
        #print(f"- Training data ({total_lines - n_testing_points} points) saved to: {training_file}")

        # Return the paths of the created files
        return testing_file, training_file

    except FileNotFoundError:
        print(f"Error: The input file '{input_file}' was not found.")
    except PermissionError:
        print(f"Error: Permission denied when trying to create or write to the output files.")
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

    # Return None values if an error occurred
    return None, None
