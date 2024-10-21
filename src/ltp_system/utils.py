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
        
        # Try saving in EPS format with error handling
        try:
            if crop:
                plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=pad_inches)
            else:
                plt.savefig('{}.eps'.format(filename))
        except Exception as e:
            logging.error(f"An error occurred while saving the EPS file: {e}")
            print(f"Could not save the EPS file due to an error: {e}")
            
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



