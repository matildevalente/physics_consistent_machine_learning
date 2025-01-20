import os
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.ticker as mticker
import matplotlib as mpl
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])
import pandas as pd
import matplotlib.gridspec as gridspec
import seaborn as sns

from matplotlib.lines import Line2D
import os
import json
import numpy as np
from pathlib import Path


from src.spring_mass_system.utils import get_predicted_trajectory, get_target_trajectory, savefig
from src.spring_mass_system.projection import get_projection_df #get_projected_trajectory



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
    'proj_NN_I': {
        'name': 'NN Projection w/ I',
        'color': '#d62728',
    },
    'proj_PINN_I': {
        'name': 'PINN Projection w/ I',
        'color':  '#9A5092'  
    }
}

state_variables_parameters = {
    'x1': {
        'key': 'x1',
        'legend': r'$x_1$',
        'units': r'$(m)$',
    },
    'v1': {
        'key': 'v1',
        'legend': r'$v_1$',
        'units': r'$(m/s)$', 
    },
    'x2': {
        'key': 'x2',
        'legend': r'$x_2$',
        'units': r'$(m)$', 
    },
    'v2': {
        'key': 'v2',
        'legend': r'$v_2$',
        'units': r'$(m/s)$',
    },
    'E': {
        'key': 'E',
        'legend': r'$E$',
        'units': r'$(J)$',
    }
}


# Compute the results to add to the plot
def get_results(config, n_time_steps, models, test_initial_states, N_initial_conditions,scaler_X,  nn_model, pinn_model, preprocessed_data, state_variables):

    # Initialize dictionaries to store accumulated results across initial states
    model_predictions_dict = {}
    rmse_dict_accumulated = {df_name: [] for df_name in models}
    mape_dict_accumulated = {df_name: [] for df_name in models}
    
    # For each initial state of the system compute the time evolution using different models
    for initial_state in tqdm(test_initial_states[:N_initial_conditions], desc="Processing initial states"):

        # 1. Get target trajectory predictions (the state variables are normalized to [-1, 1])
        df_target = get_target_trajectory(config, n_time_steps, initial_state=torch.tensor(initial_state))
        df_target_norm = _normalize_trajectory(df_target, scaler_X)
        
        # 2. Get the predictions of the NN model and PINN model as well as their projections (the state variables are normalized to [-1, 1])
        for model_name, model in [('NN', nn_model), ('PINN', pinn_model)]:

            # get NN or PINN predictions
            df = get_predicted_trajectory(config, preprocessed_data, model, n_time_steps, initial_state=torch.tensor(initial_state))
            model_predictions_dict[model_name] = _normalize_trajectory(df, scaler_X)

            # get projections
            projection_matrices = {f'{model_name}_I': torch.eye(4)}
            for name, matrix in projection_matrices.items():
                df_proj = get_projection_df(initial_state, n_time_steps, model, matrix, preprocessed_data, config, df)
                model_predictions_dict[f'proj_{name}'] = _normalize_trajectory(df_proj, scaler_X)
        
        # 3. For each model compute the RMSE and MAPE with respect to target data (normalized state variables)
        results = {}
        for col in state_variables:
            results[col] = {}
            for df_name, df in model_predictions_dict.items():
                results[col][df_name] = _compute_error_metrics(df_target_norm[col], df[col])

        # 4. Store the error values in dictionaries
        rmse_dict = {df_name: [results[col][df_name][0] for col in state_variables] for df_name in model_predictions_dict} # RMSE is on position 0 of results
        mape_dict = {df_name: [results[col][df_name][1] for col in state_variables] for df_name in model_predictions_dict} # MAPE is on position 1 of results
        
        # 5. Append the results to the accumulated dictionaries
        for df_name in model_predictions_dict:
            rmse_dict_accumulated[df_name].append(rmse_dict[df_name])
            mape_dict_accumulated[df_name].append(mape_dict[df_name])

    return mape_dict_accumulated, rmse_dict_accumulated 

# Compute MAPE and RMSE given predicted and target values 
def _compute_error_metrics(actual, predicted):
    mask = actual != 0

    # compute mape
    mape = np.mean(np.abs ((actual[mask] - predicted[mask]) / actual[mask])) * 100

    # compute rmse
    rmse = np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))
    
    return rmse, mape

# Normalize the state values of a dataframe containing a trajectory keeping the E in Joule and the time in seconds
def _normalize_trajectory(df, scaler_X):
    """
    Normalize state variables of a trajectory while preserving energy and time values.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing trajectory data with columns:
            - x1, v1, x2, v2: Positions and velocities of masses 1 and 2
            - E: Energy in Joules
            - time(s): Time in seconds
        scaler_X: Pre-fitted scaler used to normalize state variables
    
    Returns:
        pd.DataFrame: Normalized DataFrame with same structure as input where:
            - State variables (x1, v1, x2, v2) are normalized using scaler_X
            - Energy (E) and time values remain unchanged
    """

    df_norm = pd.DataFrame(
        scaler_X.transform(df[['x1', 'v1', 'x2', 'v2']].values), 
        columns=['x1', 'v1', 'x2', 'v2'], index=df.index
    )
    df_norm['E'] = df['E']
    df_norm['time(s)'] = df['time(s)']
    
    return df_norm

# For each initial state compute the improvement rate of the projection
def _analyse_proj_improvement_rate(config, mape_dict_accumulated, rmse_dict_accumulated, test_initial_states, N_initial_conditions, scaler_X):
    # study complementarity between PINN and Proj
    pairs_to_compare = [
        ["NN", "proj_NN_I", "NN", "I"], 
        ["PINN", "proj_PINN_I", "PINN", "I"], 
        ["NN", "proj_PINN_I", "PINN", "I"]       
    ]

    columns = [
        "Initial Model","Projected Model", 
        "W Matrix", "Metric", "Mean Improv (%)", "Overall Improv (%)", 
        "x1 Improv (%)", "v1 Improv (%)", "x2 Improv (%)", "v2 Improv (%)"
    ]

    df_improvement_rates = pd.DataFrame(columns= columns)
    rows = []

    for model_pair in pairs_to_compare:

        for metric_name, metric_dict in zip(["RMSE", "MAPE"], [rmse_dict_accumulated, mape_dict_accumulated]):
            
            # create counters to evalutate improvement
            overall_improvement_counter = 0
            mean_improvement_counter = 0
            x1_improv_counter, v1_improv_counter, x2_improv_counter, v2_improv_counter = 0,0,0,0
            
            model_dict = metric_dict[model_pair[0]]
            proj_dict = metric_dict[model_pair[1]]

            # evaluate each initial state and count improvement
            best_initial_condition = test_initial_states[0]
            for state_idx, initial_state in enumerate(test_initial_states[:N_initial_conditions]):
            
                # Pop the last column: we are not interested in the energy column
                err_model = np.delete(model_dict[state_idx], -1)
                err_proj  = np.delete(proj_dict [state_idx], -1)
                
                # Compute the mean
                mean_err_model = np.mean(err_model)
                mean_err_proj  = np.mean(err_proj)

                # Check if all elements in list1 are less than the corresponding elements in list2
                if all(p < m for p, m in zip(err_proj, err_model)):
                    overall_improvement_counter += 1

                    # REMOVE THIS IN THE FINAL VERSION
                    if(mean_err_proj < mean_err_model):
                        best_initial_condition = initial_state
                
                # Check improvement per state variable
                if err_proj[0] < err_model[0]:
                    x1_improv_counter += 1
                    
                if err_proj[1] < err_model[1]:
                    v1_improv_counter += 1

                if err_proj[2] < err_model[2]:
                    x2_improv_counter += 1

                if err_proj[3] < err_model[3]:
                    v2_improv_counter += 1

                if(mean_err_proj < mean_err_model):
                    mean_improvement_counter += 1

            new_row = {
                "Initial Model": model_pair[0], 
                "Projected Model": model_pair[2], 
                "W Matrix": model_pair[3],
                "Metric": metric_name, 
                "Mean Improv (%)": 100 * mean_improvement_counter / N_initial_conditions, 
                "Overall Improv (%)": 100 * overall_improvement_counter / N_initial_conditions,
                "x1 Improv (%)": 100 * x1_improv_counter / N_initial_conditions,
                "v1 Improv (%)": 100 * v1_improv_counter / N_initial_conditions,
                "x2 Improv (%)": 100 * x2_improv_counter / N_initial_conditions,
                "v2 Improv (%)": 100 * v2_improv_counter / N_initial_conditions
            }
            rows.append(new_row)
    
    # create dataframe
    df_improvement_rates= pd.DataFrame(rows, columns=columns)

    # save results to local dir in .csv format
    output_dir = config['plotting']['output_dir'] + "several_initial_conditions/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"proj_improvement_rates.csv")   
    df_improvement_rates.to_csv(save_path, index=False)
    
    #best_initial_condition_ = scaler_X.inverse_transform(best_initial_condition.reshape(1, -1))
    #print("Best Initial Condition: ", best_initial_condition_)

# Save the results that will be added to the plot
def save_results(mape_dict, rmse_dict, output_dir):
    """
    Save MAPE and RMSE results using numpy's save function.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / "mape_results.npy", mape_dict)
    np.save(output_path / "rmse_results.npy", rmse_dict)

# Load the previously ran results that will be added to the plot
def load_results(output_dir):
    """
    Load MAPE and RMSE results using numpy's load function.
    """
    output_path = Path(output_dir) 
    
    mape_dict = np.load(output_path / "mape_results.npy", allow_pickle=True).item()
    rmse_dict = np.load(output_path / "rmse_results.npy", allow_pickle=True).item()
    
    return mape_dict, rmse_dict
    
# main function to run the experiment
def plot_several_initial_conditions(config, preprocessed_data, nn_model, pinn_model, test_initial_states, n_time_steps, N_initial_conditions):

    metrics_names = ['RMSE', 'MAPE']
    scaler_X = preprocessed_data['scaler_X']
    state_variables = ['x1', 'x2', 'v1', 'v2', 'E']
    models = ['NN', 'PINN', 'proj_NN_I', 'proj_PINN_I']

    # Updated main code block
    if config['fig_3_options']['rerun_results']:
        mape_dict_accumulated, rmse_dict_accumulated = get_results(
            config, n_time_steps, models, test_initial_states, 
            N_initial_conditions, scaler_X, nn_model, pinn_model, 
            preprocessed_data, state_variables
        )
        save_results(mape_dict_accumulated, rmse_dict_accumulated, config['fig_3_options']['output_dir'])
    else:
        mape_dict_accumulated, rmse_dict_accumulated = load_results(config['fig_3_options']['output_dir'])
    
    # Analyse the improvement rates in the error metrics among the models
    _analyse_proj_improvement_rate(config, mape_dict_accumulated, rmse_dict_accumulated, test_initial_states, N_initial_conditions, scaler_X)
    
    # Create the violin plot
    _plot_violin(rmse_dict_accumulated, mape_dict_accumulated, config, models, metrics_names, state_variables)


####### ----------------------------
def  _add_mean_and_errorbar(df_plot, df_rows, metrics_names, models, axs, row, col, column):       
    
    # After drawing the violin plot, add the code to calculate and draw the mean and standard deviation 
    for metric_name in metrics_names:

        for model_idx, model in enumerate(models):
            
            # Calculate mean and std deviation for each combination of model and metric
            model_data = df_plot[(df_plot['Model'] == model) & (df_plot['Metric'] == metric_name)]['Error']
            model_mean = model_data.mean() 
            model_std = model_data.std()    
            
            # adjust the x position of the mean and std lines
            x_pos = (model_idx - 1.5) * 0.2#0.134 
            
            # Draw the standard deviation and mean lines
            if(metric_name == "RMSE"):
                axs[row][col].errorbar(x_pos, model_mean, yerr=model_std,fmt='o', color='black', capsize=5,elinewidth=1.5, markeredgewidth=2)
                axs[row][col].hlines(y=model_mean, xmin=x_pos - 0.1, xmax=x_pos + 0.1, color='black', linewidth=2, linestyle='-')

                new_row = {
                    "Model": model, 
                    "Metric": metric_name, 
                    "Output": column,
                    "Error Value": model_mean, 
                    "Std Value": model_std, 
                }
                df_rows.append(new_row)
            
            elif(metric_name == "MAPE"):
                ax = axs[2][1] if column == "E" else axs[row+1][col]
                ax.errorbar(x_pos, model_mean, yerr=model_std, fmt='o', color='black',capsize=5, elinewidth=1.5, markeredgewidth=2 )
                ax.hlines(y=model_mean, xmin=x_pos - 0.1, xmax=x_pos + 0.1, color='black', linewidth=2, linestyle='-')

                new_row = {
                    "Model": model, 
                    "Metric": metric_name, 
                    "Output": column,
                    "Error Value": model_mean, 
                    "Std Value": model_std, 
                }
                df_rows.append(new_row)

    return df_plot, df_rows

"""# function that plots the results
def _plot_violin(rmse_dict_accumulated, mape_dict_accumulated, config, models, metrics_names, state_variables):
    

    # List to map each column to the corresponding subplot position
    plot_order = [('x1', 0, 0),  ('x2', 0, 1), ('v1', 0, 3), ('v2', 0, 4), ('E', 2, 0)]
    colors = [models_parameters[model]['color'] for model in models]

    # Table 1
    columns_table = ["Model", "Metric", "Output" , "Error Value", "Std Value"]
    df_table1 = pd.DataFrame(columns= columns_table)
    df_rows = []

    # For each model get a df for each error metric
    list_df_mape, list_df_rmse = [], []
    for model_key in models:
        df_rmse = pd.DataFrame(rmse_dict_accumulated[model_key], columns=state_variables)
        df_mape = pd.DataFrame(mape_dict_accumulated[model_key], columns=state_variables)
        list_df_rmse.append(df_rmse)
        list_df_mape.append(df_mape)

    # Set up the figure with 3 rows and 2 columns
    idx_col = 0
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = gridspec.GridSpec(3, 5, width_ratios=[0.9, 0.9, 0.2, 0.9, 0.9]) 
    axs = [[fig.add_subplot(gs[row, col]) for col in range(5)] for row in range(3)]

    # loop over each of the 5 plots: x1, v1, x2, v2, E
    for column, row, col in plot_order:
        data = []
        
        for metric_idx, metric_name in enumerate(metrics_names):
            data_metric = []
            
            for model_idx, model in enumerate(models):
                model_data = [list_df_rmse, list_df_mape][metric_idx][model_idx]
                data.extend([[value, metric_name, model] for value in model_data[column]])
                data_metric.extend([[value, metric_name, model] for value in model_data[column]])
                
            if(metric_name == "RMSE"):
                # get rmse data
                df_plot_rmse = pd.DataFrame(data_metric, columns=['Error', 'Metric', 'Model'])
                # Draw the RMSE plot
                ax = axs[row][col]
                # plot
                sns.violinplot(x='Metric', y='Error', hue='Model', data=df_plot_rmse, scale='count', inner='point', dodge=True, saturation=1, palette=colors, ax=ax)
                # formatting
                ax.set_title(state_variables_parameters[column]['legend'], fontsize = 20)
                # y label
                ax.set_ylabel('RMSE ', fontsize = 18)
                ax.tick_params(axis='y', labelsize=18)  
                ax.set_yscale('log') if column == "E" else None
                # x label
                ax.set_xlabel(None)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                # legend & plot
                ax.get_legend().remove() 
                
 
            elif(metric_name == "MAPE"):
                # get mape data
                df_plot_mape = pd.DataFrame(data_metric, columns=['Error', 'Metric', 'Model'])
                # Draw the MAPE plot    
                if(column == "E"):            
                    ax = axs[2][1]
                    ax.set_yscale('log')
                else:
                    ax = axs[row+1][col]
                # plot
                sns.violinplot(x='Metric', y='Error', hue='Model', data=df_plot_mape, scale='count', inner='point', dodge=True, saturation=1, palette=colors, ax=ax)
                # formatting
                ax.set_title(state_variables_parameters[column]['legend'], fontsize = 20) 
                # y label
                ax.set_ylabel('MAPE (\%)', fontsize=18)
                ax.tick_params(axis='y', labelsize=18)
                # x label
                ax.set_xlabel(None)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                # legend 
                ax.get_legend().remove()
                
                
        df_plot = pd.DataFrame(data, columns=['Error', 'Metric', 'Model'])
        df_plot, df_rows = _add_mean_and_errorbar(df_plot, df_rows, metrics_names, models, axs, row, col, column)
        idx_col += 1
    
    # create dataframe & save results to local dir in .csv format
    df_table1 = pd.DataFrame(df_rows, columns=columns_table)
    file_path = "output/spring_mass_system/proj_improvement_rates/Table1.csv"
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    df_table1.to_csv(file_path, index=False)
    
    # remove ylabels and yticks to simplify plot
    for row, col in [[0, 1], [0, 4], [1, 1], [1, 4]]:
        axs[row][col].set_ylabel(None)
        #axs[row][col].tick_params(axis='y', labelleft=False) 
    
    # hide axis
    for row, col in [[2, 2], [2, 3], [2, 4], [0, 2], [1, 2]]:
        axs[row][col].axis('off')

    # Apply log scale to specific y-axes
    #for row, col in [[1, 0], [1, 1], [1, 3], [1, 4]]:
    #    axs[row][col].set_yscale('log')

    #axs[1][0].set_ylim(-1, 2.5)
    #axs[1][1].set_ylim(-1.5, 5)
    #axs[1][3].set_ylim(-1, 2.5)
    #axs[1][4].set_ylim(-2.1, 5)

    # remove ylabels and yticks to simplify plot
    #for row, col in [[0, 1], [0, 4]]:
    #    axs[row][col].tick_params(axis='y', labelleft=False) 

    # fix y ranges
    #for ax1, ax2 in [[[0, 0], [0, 1]],[[0, 3], [0, 4]],[[1, 0], [1, 1]],[[1, 3], [1, 4]]]:
    #for ax1, ax2 in [[[0, 0], [0, 1]],[[0, 3], [0, 4]]]:
    #    y_min_widest = min(axs[ax1[0]][ax1[1]].get_ylim()[0], axs[ax2[0]][ax2[1]].get_ylim()[0])
    #    y_max_widest = max(axs[ax1[0]][ax1[1]].get_ylim()[1], axs[ax2[0]][ax2[1]].get_ylim()[1])
    #    axs[ax1[0]][ax1[1]].set_ylim(y_min_widest, y_max_widest)
    #    axs[ax2[0]][ax2[1]].set_ylim(y_min_widest, y_max_widest)
    


    # Move the y-axis label and y-ticks to the right side for axs[2][1]
    axs[2][1].yaxis.set_label_position("right")  
    axs[2][1].tick_params(axis='y', labelright=True, labelleft=False)  
    axs[2][1].set_yticks([1e0, 1e-1, 1e-3, 1e-6])  
    axs[2][1].set_yticklabels([r'$10^{0}$', r'$10^{-1}$', r'$10^{-3}$', r'$10^{-6}$'])  
    axs[2][0].set_yticks([1e0, 1e-1, 1e-3, 1e-6])  
    axs[2][0].set_yticklabels([r'$10^{0}$',r'$10^{-1}$', r'$10^{-3}$', r'$10^{-6}$'])  


    # Merge cells [2,3] and [2,4] 
    gs = gridspec.GridSpec(3, 5, figure=fig) 
    ax_legend = fig.add_subplot(gs[2, 3:5])  
    ax_legend.axis('off')  # Hide the axis

    # Add the legend in the merged cell
    ax_legend.legend(
        handles=[
            plt.Line2D([0], [0], color=colors[0], lw=12, label='NN'),
            plt.Line2D([0], [0], color=colors[1], lw=12, label='PINN'),
            plt.Line2D([0], [0], color=colors[2], lw=12, label=r'NN Projection'),
            plt.Line2D([0], [0], color=colors[3], lw=12, label=r'PINN Projection'),
            Line2D([0], [0], color='black', marker='_', markersize=7, linestyle='', markeredgewidth=2, label='Mean ± Std'),
        ],
        loc='center', 
        bbox_to_anchor=(0.43, 0.5),
        frameon=False,
        prop={'size': 22},  
        markerscale=1.5,
        labelspacing=0.45,  # vertical spacing between rows 
        ncol=2,
        columnspacing=1.7  # Increase this value for more space between columns
    )

    # Save figure
    output_dir = config['plotting']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_2")
    plt.savefig('{}.pdf'.format(save_path), bbox_inches='tight', pad_inches=0.2)


"""
# function that plots the results
def _plot_violin(rmse_dict_accumulated, mape_dict_accumulated, config, models, metrics_names, state_variables):
    

    # List to map each column to the corresponding subplot position
    plot_order = [('x1', 0, 0),  ('x2', 0, 1), ('v1', 1, 0), ('v2', 1, 1), ('E', 2, 0)]
    colors = [models_parameters[model]['color'] for model in models]

    # Table 1
    columns_table = ["Model", "Metric", "Output" , "Error Value", "Std Value"]
    df_table1 = pd.DataFrame(columns= columns_table)
    df_rows = []

    # For each model get a df for each error metric
    list_df_rmse = []
    for model_key in models:
        df_rmse = pd.DataFrame(rmse_dict_accumulated[model_key], columns=state_variables)
        list_df_rmse.append(df_rmse)

    # Set up the figure with 3 rows and 2 columns
    idx_col = 0
    mpl.use('pgf')
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, width_ratios=[0.8, 0.8]) 
    axs = [[fig.add_subplot(gs[row, col]) for col in range(2)] for row in range(3)]

    # loop over each of the 5 plots: x1, v1, x2, v2, E
    for column, row, col in plot_order:
        data = []
        
        for metric_idx, metric_name in enumerate(['RMSE']):
            data_metric = []
            
            for model_idx, model in enumerate(models):
                model_data = [list_df_rmse][metric_idx][model_idx]
                data.extend([[value, metric_name, model] for value in model_data[column]])
                data_metric.extend([[value, metric_name, model] for value in model_data[column]])
                
            if(metric_name == "RMSE"):
                # get rmse data
                df_plot_rmse = pd.DataFrame(data_metric, columns=['Error', 'Metric', 'Model'])
                # Draw the RMSE plot
                ax = axs[row][col]
                # plot
                sns.violinplot(x='Metric', y='Error', hue='Model', data=df_plot_rmse, scale='count', inner='point', dodge=True, saturation=1, palette=colors, ax=ax)
                # formatting
                ax.set_title(state_variables_parameters[column]['legend'], fontsize = 30)
                # y label
                ax.set_ylabel('RMSE ', fontsize = 24)
                ax.tick_params(axis='y', labelsize=24)  
                ax.set_yscale('log') if column == "E" else None
                # x label
                ax.set_xlabel(None)
                ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                # legend & plot
                ax.get_legend().remove() 
                
                
                
        df_plot = pd.DataFrame(data, columns=['Error', 'Metric', 'Model'])
        df_plot, df_rows = _add_mean_and_errorbar(df_plot, df_rows, metrics_names, models, axs, row, col, column)
        idx_col += 1
    
    # create dataframe & save results to local dir in .csv format
    df_table1 = pd.DataFrame(df_rows, columns=columns_table)
    output_dir = config['plotting']['output_dir'] + "several_initial_conditions/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"errors_norm_trajectories_Figure_3.csv")   
    df_table1.to_csv(save_path, index=False)


    # remove ylabels and yticks to simplify plot
    for row, col in [[0, 1], [1, 1]]:
        axs[row][col].set_ylabel(None)

    for row, col in [[0, 0], [0, 1], [1, 0], [1, 1]]:
        axs[row][col].set_ylim(0, 0.08)  
        axs[row][col].set_yticks([0.0, 0.02, 0.04, 0.06])  

        # Set the y-axis to display values in scientific notation
        axs[row][col].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x * 100:.0f}'))

        # Set the y-axis label with a scaling factor
        axs[row][col].text(-0.19, 1, r'$ (\times 10^{-2})$', transform=axs[row][col].transAxes,verticalalignment='center', horizontalalignment='left', fontsize=20)

    # hide axis
    for row, col in [[2, 1]]:
        axs[row][col].axis('off')

    # Move the y-axis label and y-ticks to the right side for axs[2][1]
    axs[2][1].yaxis.set_label_position("right")  
    axs[2][1].tick_params(axis='y', labelright=True, labelleft=False)  
    axs[2][1].set_yticks([1, 1e-2, 1e-4, 1e-6])  
    axs[2][1].set_yticklabels([r'$1$', r'$10^{-2}$', r'$10^{-4}$', r'$10^{-6}$'])  
    axs[2][0].set_yticks([1, 1e-2, 1e-4, 1e-6])  
    axs[2][0].set_yticklabels([r'$1$',r'$10^{-2}$', r'$10^{-4}$', r'$10^{-6}$'])  

    # Adjust GridSpec to 3 rows and 5 columns as before
    gs = gridspec.GridSpec(3, 5, figure=fig)

    # Place the legend in cell (2,1) without merging cells
    ax_legend = fig.add_subplot(gs[2, 1])  
    ax_legend.axis('off')  # Hide the axis

    # Add the legend in the specified cell
    ax_legend.legend(
        handles=[
            plt.Line2D([0], [0], color=colors[0], lw=12, label='NN'),
            plt.Line2D([0], [0], color=colors[1], lw=12, label='PINN'),
            plt.Line2D([0], [0], color=colors[2], lw=12, label=r'NN projection'),
            plt.Line2D([0], [0], color=colors[3], lw=12, label=r'PINN projection'),
            Line2D([0], [0], color='black', marker='+', markersize=25, linestyle='', markeredgewidth=2, label='Mean ± Std'),
        ],
        loc='center', 
        bbox_to_anchor=(3.35, 0.5),
        frameon=False,
        prop={'size': 24},  
        markerscale=1.5,
        labelspacing=0.45,  # vertical spacing between rows 
        ncol=1,
        columnspacing=1.7  # Increase this value for more space between columns
    )


    # Save figure
    output_dir = config['plotting']['output_dir'] + "several_initial_conditions/"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"Figure_3")
    savefig(save_path, pad_inches=0.2)
    #plt.savefig('{}.pdf'.format(save_path),bbox_inches='tight' , pad_inches=0.2)





