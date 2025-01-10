import os
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import matplotlib as mpl
import matplotlib.ticker as ticker

from typing import Dict, Any
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.patches import Circle, ConnectionPatch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


from src.spring_mass_system.utils import figsize, newfig, savefig, compute_total_energy

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino"],  # LaTeX-compatible serif fonts
    "font.monospace": ["Courier"],      # LaTeX-compatible monospace fonts
}

plt.rcParams.update(pgf_with_latex)


models_parameters = {
    'NN': {
        'name': 'NN',
        'color': '#0b63a0',
    },
    'PINN': {
        'name': 'PINN',
        'color': '#FFAE0D'  
    },
    'proj_nn': {
        'name': 'NN Projection',
        'color': '#d62728'  
    },
    'proj_pinn': {
        'name': 'PINN Projection',
        'color': '#9A5092'  
    }
}

# compute the rmse and the sem values
def calculate_metrics(actual, predicted):
    
    # compute rmse error
    mask = actual != 0
    rmse = np.sqrt(np.mean((actual[mask] - predicted[mask])**2))
    sem = np.std(actual[mask] - predicted[mask]) / np.sqrt(len(actual[mask]))

    return rmse, sem

# get the results for the plots
def get_results(scaler_X, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn):
    # normalize the state variable values of the target and predicted trajectories
    df_target_norm = pd.DataFrame(scaler_X.transform(df_target[['x1', 'v1', 'x2', 'v2']].values), columns=['x1', 'v1', 'x2', 'v2'], index=df_target.index)
    df_nn_norm = pd.DataFrame(scaler_X.transform(df_nn[['x1', 'v1', 'x2', 'v2']].values), columns=['x1', 'v1', 'x2', 'v2'], index=df_nn.index)
    df_pinn_norm = pd.DataFrame(scaler_X.transform(df_pinn[['x1', 'v1', 'x2', 'v2']].values), columns=['x1', 'v1', 'x2', 'v2'], index=df_pinn.index)
    df_proj_nn_norm = pd.DataFrame(scaler_X.transform(df_proj_nn[['x1', 'v1', 'x2', 'v2']].values), columns=['x1', 'v1', 'x2', 'v2'], index=df_proj_nn.index)
    df_proj_pinn_norm = pd.DataFrame(scaler_X.transform(df_proj_pinn[['x1', 'v1', 'x2', 'v2']].values), columns=['x1', 'v1', 'x2', 'v2'], index=df_proj_pinn.index)

    # keep the energy value in Joule (not normalized)
    df_target_norm['E'] = df_target['E']
    df_nn_norm['E'] = df_nn['E']
    df_pinn_norm['E'] = df_pinn['E']
    df_proj_nn_norm['E'] = df_proj_nn['E']
    df_proj_pinn_norm['E'] = df_proj_pinn['E']

    # compute the mape and sem of the predicted state variables with respect to the target values
    dataframes = {'NN': df_nn_norm, 'PINN': df_pinn_norm, 'proj_nn': df_proj_nn_norm, 'proj_pinn': df_proj_pinn_norm}
    results = {}
    for idx, col in enumerate(['x1', 'v1', 'x2', 'v2']):
        results[col] = {}
        for df_name, df in dataframes.items():
            results[col][df_name] = calculate_metrics(df_target_norm[col], df[col])
    
    # compute the mape and sem in compliance with the physical laws:
    #    for each predicted trajectory, use the energy of the first predicted state as a target
    results['E'] = {}
    for df_name, df in dataframes.items():
        # Create a new series with the same length, filled with the initial state energy
        initial_energy_series = pd.Series([df['E'][0]] * len(df['E']))
        results['E'][df_name] = calculate_metrics(initial_energy_series, df['E'])
    
    # compute the Runge Kutta compliance with the energy conservation law
    initial_energy_series = pd.Series([df_target['E'][0]] * len(df_target['E']))
    results['E']['RK'] = calculate_metrics(initial_energy_series, df_target['E'])
    rk_energy_mae = np.mean(np.abs(df_target['E'] - df_target['E'][0]))
    rk_energy_mape = np.mean(np.abs(df_target['E'] - df_target['E'][0])/np.abs(df_target['E'][0]))*100
    print(f"RK Energy MAE = {rk_energy_mae:.4e}    RK Energy RMSE = {results['E']['RK'][0]:.4e}    RK Energy MAPE = {rk_energy_mape:.4f} %")
    
    # create a dict for the mape values and the sem values
    mape_dict = {df_name: [results[col][df_name][0] for col in ['x1', 'v1', 'x2', 'v2', 'E']] for df_name in dataframes}
    sem_dict = {df_name: [results[col][df_name][1] for col in ['x1', 'v1', 'x2', 'v2', 'E']] for df_name in dataframes}
    # append the values for the RK
    mape_dict['RK'] = [results['E']['RK'][0]]
    sem_dict['RK'] = [results['E']['RK'][1]]

    return mape_dict, sem_dict

# save .csv file with the error values presented in the plot
def save_df(variables, mape_dict, sem_dict):
    
    def append_rows(data_dict, metric, rows, variables):
        
        for model, values in data_dict.items():

            if(model == "RK"):
                for variable, value in zip(variables, values):
                    rows.append({"Model": "RK","Variable": "E","Metric": metric,"Value": value})
            else:
                for variable, value in zip(variables, values):
                    rows.append({
                        "Model": model,
                        "Variable": variable,
                        "Metric": metric,
                        "Value": value
                    })

    # IMPLEMENT CODE HERE #################################
    rows = []
    # append the RMSE values
    append_rows(mape_dict, "RMSE", rows, variables)
    # append the SEM values
    append_rows(sem_dict, "SEM", rows, variables)
    # save dataframe
    df = pd.DataFrame(rows)
    #######################################################

    # Save .csv file with the error values presented in the plot
    output_dir = "output/spring_mass_system/single_initial_condition/"
    os.makedirs(output_dir, exist_ok=True)  
    csv_path = os.path.join(output_dir, "error_values.csv")
    df.to_csv(csv_path, index=False)

# Figure 2d: Bar plot of the MAPE of the NN, standard PINN, and corresponding projected trajectories
def plot_bar_plot(config, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn, preprocessed_data):
    
    # Separate the indices for different groups
    variables_indices = np.arange(4)  
    energy_index = 4  
    state_variable_names = ['$x_1$', '$v_1$', '$x_2$', '$v_2$']
    variables = ["x1", "v1", "x2", "v2", "E"]
    scaler_X = preprocessed_data['scaler_X']

    # Get the error values to add to the plot
    mape_dict, sem_dict = get_results(scaler_X, df_target, df_nn, df_pinn, df_proj_nn, df_proj_pinn)
    save_df(variables, mape_dict, sem_dict)

    # Create the plot with the computed error values
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), gridspec_kw={'width_ratios': [2.5, 1], 'height_ratios': [1]})
    
    ########################################### Create bar plot for the state variables
    bar_width = 0.2 
    max_mape_with_yerr = -float('inf')
    
    for i, (df_name, mape) in enumerate(mape_dict.items()):
        if(df_name == "RK"):
            continue
        
        mape_values = mape[:4]
        sem_values = sem_dict[df_name][:4]

        # add the 4 bars, one for each state variable
        ax1.bar(variables_indices + (i - (len(mape_dict) / 2)) * bar_width + 0.2, 
            mape_values, bar_width, yerr=sem_values, label=models_parameters[df_name]['name'], 
            color=models_parameters[df_name]['color'], capsize=5, error_kw={'capthick': 2}
        )
        
        for m, yerr in zip(mape_values, sem_values):
            max_mape_with_yerr = max(max_mape_with_yerr, m + yerr)
    
    # format x axis
    ax1.set_xticks(variables_indices+ (bar_width / 2))
    ax1.set_xticklabels(state_variable_names, fontsize=70)
    # format y axis    
    ax1.set_ylabel('RMSE', fontsize=30, labelpad=10)
    y_ticks = np.round(np.linspace(0, max_mape_with_yerr , 5), 3)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([f"{(y_tick)}" for y_tick in y_ticks], fontsize=55)
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(3, 4))
    ax1.yaxis.get_offset_text().set_text(f"(x10^{int(np.log10(max_mape_with_yerr))})")
    ax1.yaxis.get_offset_text().set_fontsize(40)
    ax1.yaxis.get_offset_text().set_position((-0.05, 0))
    ax1.tick_params(axis='both', which='major', labelsize=65)
    plt.draw()
    ######################################################################################

    ########################################### Create bar plot for the energy conservation errors
    bar_width = 0.6
    for i, (df_name, mape) in enumerate(mape_dict.items()):
        if(df_name == "RK"):
            continue
        ax2.bar(
            energy_index + (i-1) * bar_width, 
            mape[4], 
            bar_width, 
            yerr=sem_dict[df_name][4], 
            label=models_parameters[df_name]['name'], 
            color=models_parameters[df_name]['color'], 
            capsize=5, 
            error_kw={'capthick': 2}
        )
    # format y axis
    ax2.set_yscale('log')
    #ax2.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=4))
    #ax2.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    ax2.set_yticks([1e-1, 1e-3, 1e-5, 1e-7])
    ax2.set_yticklabels([r'$10^{-1}$', r'$10^{-3}$', r'$10^{-5}$', r'$10^{-7}$'], fontsize=30)
    ax2.tick_params(axis='y', labelsize=30)
    ax2.set_ylabel('RMSE (J)', fontsize=30)
    # format x axis
    ax2.set_xticks([(energy_index + 0.3)])
    ax2.set_xticklabels(['$\mathit{E}$'], fontsize=40)
    ax2.set_position([0.83, 0.1, 0.26, 0.45])  # Left, bottom, width, height
    ######################################################################################


    # Move the legend into the empty space between the plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=26, loc='center right', bbox_to_anchor=(1.12, 0.80), frameon=True,
            handletextpad=0.5,  # Reduce the space between the handle and the text
            labelspacing=0.1)   # Reduce the vertical space between the labels

    # Save figure
    output_dir = config['plotting']['output_dir'] + "single_initial_condition/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"barplot.svg"), bbox_inches='tight', pad_inches=0.2)
    plt.savefig(os.path.join(output_dir, f"barplot.pdf"), bbox_inches='tight', pad_inches=0.2)

  


