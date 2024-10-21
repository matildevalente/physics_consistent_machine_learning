import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.ltp_system.utils import set_seed, load_dataset, load_config
from src.ltp_system.utils import savefig
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

from src.ltp_system.data_prep import DataPreprocessor, LoadDataset
from src.ltp_system.pinn_nn import get_trained_bootstraped_models, load_checkpoints, NeuralNetwork, get_average_predictions, get_predictive_uncertainty
from src.ltp_system.projection import get_average_predictions_projected,constraint_p_i_ne

# load and preprocess the test dataset from a local directory
def load_data(test_filename, preprocessed_data):
    # load and extract experimental dataset
    test_dataset = LoadDataset(test_filename)
    test_targets, test_inputs = test_dataset.y, test_dataset.x

    # apply log transform to the skewed features
    if len(preprocessed_data.skewed_features_in) > 0:
        test_inputs[:, preprocessed_data.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, preprocessed_data.skewed_features_in]))

    if len(preprocessed_data.skewed_features_out) > 0:
        test_targets[:, preprocessed_data.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, preprocessed_data.skewed_features_out]))

    # 3. normalize targets with the model used on the training data
    normalized_inputs  = torch.cat([torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) for i, scaler in enumerate(preprocessed_data.scalers_input)], dim=1)
    normalized_targets = torch.cat([torch.from_numpy(scaler.transform(test_targets[:, i:i+1])) for i, scaler in enumerate(preprocessed_data.scalers_output)], dim=1)
 

    return normalized_inputs, normalized_targets

def generate_p_inputs(preped_data, normalized_inputs_):
    
    N_points = 1000
    p_max    = torch.max(torch.tensor(normalized_inputs_[:,0]))
    p_min    = torch.min(torch.tensor(normalized_inputs_[:,0]))
    i_fixed  = normalized_inputs_[:,1][1]
    R_fixed  = normalized_inputs_[:,2][1]

    step = (p_max - p_min) / (N_points - 1)
    input_data = np.array([[p_min + i * step, i_fixed , R_fixed ] for i in range(N_points)])
    
    return torch.tensor(input_data)

# compute the mape uncertainty of the aggregated nn models
def get_mape_uncertainty(normalized_targets, normalized_model_pred_uncertainty):
    # normalized_model_pred_uncertainty is the std / sqrt(n_models) of each model prediction
    # normalized_targets is the corresponding model target
    # here we compute the uncertaity of the mape by propagating the uncertainty of each model prediction

    # Compute MAPE uncertainty based on individual prediction uncertainties
    n = len(normalized_targets)
    mape_uncertainty = 0
    
    # Loop over each sample
    for i in range(n):
        # Extract the target row and uncertainty row for the current sample
        target_row = normalized_targets[i]
        uncertainty_row = normalized_model_pred_uncertainty[i]

        # Ensure that the target row has no zeros to avoid division errors
        non_zero_mask = target_row != 0  # Boolean mask where targets are non-zero

        # Use the mask to compute the uncertainty for valid (non-zero target) values
        mape_uncertainty += torch.sum((uncertainty_row[non_zero_mask] / target_row[non_zero_mask]) ** 2).item()

    # Final uncertainty
    mape_uncertainty = np.sqrt(mape_uncertainty) / n

    return mape_uncertainty

# the the mape and mape uncertainty of the nn aggregated model
def evaluate_model(normalized_model_predictions, normalized_targets, normalized_model_pred_uncertainty):
    
    # compute the mape and sem with respect to target
    mape = mean_absolute_percentage_error(normalized_targets, normalized_model_predictions)
    mape_uncertainty = get_mape_uncertainty(normalized_targets, normalized_model_pred_uncertainty)
    rmse = np.sqrt(mean_squared_error(normalized_targets, normalized_model_predictions))

    return mape, mape_uncertainty, normalized_model_predictions, rmse

# get the mape of the nn projected predictions
def evaluate_projection(normalized_proj_predictions, normalized_targets):

    # compute the mape and sem with respect to target
    mape = mean_absolute_percentage_error(normalized_targets, normalized_proj_predictions)
    rmse = np.sqrt(mean_squared_error(normalized_targets, normalized_proj_predictions))

    return mape, rmse



def get_data_Figure_6b(config, networks, file_path, w_matrix):
  
    # /// 1. EXTRACT DATASET & PREPROCESS THE DATASET///
    _, full_dataset = load_dataset(config, file_path)
    
    preped_data = DataPreprocessor(config)
    preped_data.setup_dataset(full_dataset.x, full_dataset.y)  
    
    normalized_inputs, normalized_targets = load_data(file_path, preped_data)

    # get the normalized model predictions
    normalized_inputs_ = normalized_inputs.clone() 
    normalized_model_predictions_simul =  get_average_predictions(networks, normalized_inputs_)
    normalized_model_pred_uncertainty_simul =  get_predictive_uncertainty(networks, normalized_inputs_) 
    normalized_proj_predictions_simul  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions_simul), normalized_inputs_, preped_data, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions_simul = torch.tensor(np.stack(normalized_proj_predictions_simul))

    # generate constant pressure inputs 
    normalized_inputs_contiuous_p = generate_p_inputs(preped_data, normalized_inputs_)
    
    # compute 
    normalized_model_predictions_contiuous_p =  get_average_predictions(networks, normalized_inputs_contiuous_p)
    normalized_model_pred_uncertainty_contiuous_p =  get_predictive_uncertainty(networks, normalized_inputs_contiuous_p)
    normalized_proj_predictions_contiuous_p = get_average_predictions_projected(normalized_model_predictions_contiuous_p, normalized_inputs_contiuous_p, preped_data, constraint_p_i_ne, w_matrix) 
    normalized_proj_predictions_contiuous_p = torch.tensor(np.stack(normalized_proj_predictions_contiuous_p))

    # Inverse Normalization and Log Transform
    denormalized_inputs_simul, denormalized_targets_simul = preped_data.inverse_transform(normalized_inputs_, normalized_targets)
    _, denormalized_proj_predictions_simul          = preped_data.inverse_transform(normalized_inputs_, normalized_proj_predictions_simul)
    _, denormalized_model_predictions_simul         = preped_data.inverse_transform(normalized_inputs_, normalized_model_predictions_simul)
    #
    denormalized_inputs_contiuous_p, denormalized_model_predictions_contiuous_p = preped_data.inverse_transform(normalized_inputs_contiuous_p, normalized_model_predictions_contiuous_p)
    denormalized_inputs_contiuous_p, denormalized_proj_predictions_contiuous_p = preped_data.inverse_transform(normalized_inputs_contiuous_p, normalized_proj_predictions_contiuous_p)


    #
    mape_nn, mape_uncertainty_nn, normalized_model_predictions_nn, rmse_nn = evaluate_model(normalized_targets, normalized_model_predictions_simul, normalized_model_pred_uncertainty_simul)
    mape_proj, rmse_proj = evaluate_projection(normalized_proj_predictions_simul, normalized_targets)
    
    denormalized_predictions_dict = {
       'discrete_inputs'             : denormalized_inputs_simul, 
       'discrete_targets'            : denormalized_targets_simul,
       'discrete_nn_predictions'     : denormalized_model_predictions_simul,
       'discrete_nn_proj_predictions': denormalized_proj_predictions_simul,

       'const_p_inputs'              : denormalized_inputs_contiuous_p, 
       'nn_model_outputs'            : denormalized_model_predictions_contiuous_p, 
       'nn_model_pred_uncertainties' : normalized_model_pred_uncertainty_contiuous_p, 
       'nn_proj_outputs'             : denormalized_proj_predictions_contiuous_p, 
    }
    errors_dict = {
       'mape_nn':mape_nn, 
       'rmse_nn': rmse_nn, 
       'mape_proj':mape_proj,
       'rmse_proj':rmse_proj
    }
    """#     plots
    norm_or_not = 0
    mean_rse_arr_NN, mean_rse_arr_PINN = [],[] 
    for i in range(len(preped_data.output_features)):
        #title = f"Modelling vs. Target (I= {round(i_fixed*1000, 0)} mA R= {R_fixed:.1e} m)"
        mean_rse, max_rse, mean_rse_c, max_rse_c = PhysicalPlot_PINN_NN(input_data[:,0], output_data[:,i], output_data_contraint[:,i], output_data_norm_error[:,i], path_physical, data_prepocessed.output[i], " ", input_data_exp[:,0], output_data_exp[:, i], output_data_exp_pred[:, i], output_data_exp_pred_contraint[:, i], "NN", "P (Pa)", data_prepocessed.output[i], norm_or_not, log_y_scale)
        #mean_rse_c, max_rse_c = PhysicalPlot(input_data[:,0], output_data_contraint[:,i], output_data_norm_error[:,i], path_physical, data_prepocessed.output[i], title, input_data_exp[:,0], output_data_exp[:, i], output_data_exp_pred_contraint[:, i], "PINN", "P (Pa)", data_prepocessed.output[i], norm_or_not, log_y_scale)
        mean_rse_arr_NN.append(mean_rse)
        mean_rse_arr_PINN.append(mean_rse_c)"""

    return denormalized_predictions_dict, errors_dict

# physical plot - outputs vs. pressure at a given discharge current (1 plot)
def Figure_6b(config, denormalized_predictions_dict, errors_dict):
  
    outputs = config['dataset_generation']['output_features']

    # inputs and targets based on the given file (simulation points)
    discrete_inputs=  denormalized_predictions_dict['discrete_inputs']
    discrete_targets=  denormalized_predictions_dict['discrete_targets']
    discrete_nn_predictions = denormalized_predictions_dict['discrete_nn_predictions']
    discrete_nn_proj_predictions = denormalized_predictions_dict['discrete_nn_proj_predictions']

    # inputs and targets continuous distribution
    const_p_inputs   =  denormalized_predictions_dict['const_p_inputs']
    nn_model_outputs =  denormalized_predictions_dict['nn_model_outputs']
    nn_proj_outputs  =  denormalized_predictions_dict['nn_proj_outputs']
    nn_model_pred_uncertainties = denormalized_predictions_dict['nn_model_pred_uncertainties']


    for i, output in enumerate(outputs):
        mape_nn = mean_absolute_percentage_error(discrete_targets[:,i], discrete_nn_predictions[:,i])
        mape_nn_proj = mean_absolute_percentage_error(discrete_targets[:,i], discrete_nn_proj_predictions[:,i])

        plt.clf()
        plt.style.use(['science','nature'])
        # Set font to sans-serif
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica']

        # Plot 1: Scatter plot comparing ne_model and calculated ne
        fig, ax = plt.subplots(figsize=(6, 5))
        # Tick params
        ax.legend(fontsize='large', loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=15, width=2)
        ax.tick_params(axis='both', which='minor', labelsize=15, width=2)
        # Ensure all ticks on the x- and y- axes are labeled
        ax.xaxis.set_tick_params(which='both', direction='in', top=True, bottom=True)
        ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
        # Adjust the font size of the order of magnitude in the plot
        ax.yaxis.get_offset_text().set_fontsize(16)


        ax.errorbar(const_p_inputs[:,0], nn_model_outputs[:,i], yerr=None, xerr=nn_model_pred_uncertainties[:,i], fmt='ro', color='blue', ecolor='lightblue', capsize=4, markersize=3)
        ax.plot([], [], '-', color='blue', markersize=5, label=f'{output}, NN, mean error = {mape_nn:.3f}%', linewidth=3)
        ax.errorbar(const_p_inputs[:,0], nn_proj_outputs[:,i], yerr=None, xerr=nn_model_pred_uncertainties[:,i], fmt='ro', color='green', ecolor='lightgreen', capsize=4, markersize=3)
        ax.plot([], [], '--', color='green', markersize=5, label=f'{output}, NN Projection, mean error = {mape_nn_proj:.3f}%', linewidth=3)
        ax.plot(discrete_inputs[:,0], discrete_targets[:,i], 'o', color='red', label=f'{output}, Simulation', markersize=6, zorder=10)

        
        ax.legend(fontsize="10")
        ax.set_xlabel("Pressure (Pa)", fontsize=14, fontweight='bold')
        ax.set_ylabel(output, fontsize=14, fontweight='bold')
        #ax.set_title(title)
        #ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)

        # Make axis numbers bold
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(12)
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2.5)

        # Save figures
        output_dir = config['plotting']['output_dir'] 
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"Figure_6b_{output}.pdf")
        plt.savefig(save_path, pad_inches = 0.2)

# physical plot
def plot_output_vs_pressure_multi_i(data_prepocessed, dataset_names, networks, num_models, path):

  err_out = ["err_" + s for s in data_prepocessed.output]
  df_plot = pd.DataFrame(columns=["exp_or_model"]+data_prepocessed.input + data_prepocessed.output + err_out)
  input_data_total, output_data_total, input_data_exp_total, output_data_exp_total, output_data_exp_pred_total,output_data_norm_error_total = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
  for dataset_name in dataset_names:
    # 1. Extract experimental dataset
    full_dataset_exp = LoadDataset(dataset_name)
    output_data_exp, input_data_exp = full_dataset_exp.y, full_dataset_exp.x
    
    # 2. Apply log transform to the skewed features
    if len(data_prepocessed.skewed_features_in) > 0:
        input_data_exp[:, data_prepocessed.skewed_features_in] = torch.log1p(torch.tensor(input_data_exp[:, data_prepocessed.skewed_features_in]))

    if len(data_prepocessed.skewed_features_out) > 0:
        output_data_exp[:, data_prepocessed.skewed_features_out] = torch.log1p(torch.tensor(output_data_exp[:, data_prepocessed.skewed_features_out]))

    # 3. normalize experimental points with the model used on the training data
    input_data_exp_norm = torch.cat([torch.from_numpy(scaler.transform(input_data_exp[:, i:i+1])) for i, scaler in enumerate(data_prepocessed.scalers_input)], dim=1)
    output_data_exp_norm = torch.cat([torch.from_numpy(scaler.transform(output_data_exp[:, i:i+1])) for i, scaler in enumerate(data_prepocessed.scalers_output)], dim=1)
    output_data_exp_norm_pred_contraint =  get_average_predictions(networks, torch.tensor(input_data_exp_norm), torch.tensor(num_models), len(data_prepocessed.output))

    # 4. generate sample
    N_points, p_max, p_min, i_fixed, R_fixed = 2000, torch.max(torch.tensor(input_data_exp[:,0])), torch.min(torch.tensor(input_data_exp[:,0])), input_data_exp[:,1][1], input_data_exp[:,2][1]
    step = (p_max - p_min) / (N_points - 1)
    input_data = [[p_min + i * step, i_fixed , R_fixed ] for i in range(N_points)]
    #     normalize sample with the model used on the training data
    input_data = np.array(input_data)
    input_data_norm = torch.cat([torch.from_numpy(scaler.transform(input_data[:, i:i+1])) for i, scaler in enumerate(data_prepocessed.scalers_input)], dim=1)
    input_data_norm = input_data_norm.detach().numpy()
    #     evaluate
    output_data_norm =  get_average_predictions(networks, torch.tensor(input_data_norm), num_models, len(data_prepocessed.output))
    output_data_norm_error = get_predictive_uncertainty(networks, torch.tensor(input_data_norm), num_models)

    #EVALUATE WITHOUT NORMALIZATION - REVERT NORMALIZATION
    input_data, output_data = data_prepocessed.inverse_transform(input_data_norm, output_data_norm)
    input_data_exp, output_data_exp = data_prepocessed.inverse_transform(input_data_exp_norm, output_data_exp_norm)
    input_data_exp, output_data_exp_pred = data_prepocessed.inverse_transform(input_data_exp_norm, output_data_exp_norm_pred_contraint)
    #
    input_data[:, 1] = np.round(input_data[:, 1], 2)
    input_data_exp[:, 1] = np.round(input_data_exp[:, 1], 2)
    # concat
    input_data_total = input_data.numpy() if len(input_data_total) == 0 else np.concatenate((input_data_total, input_data.numpy()))
    output_data_total = output_data.numpy() if len(output_data_total) == 0 else np.concatenate((output_data_total, output_data.numpy()))
    input_data_exp_total = input_data_exp.numpy() if len(input_data_exp_total) == 0 else np.concatenate((input_data_exp_total, input_data_exp.numpy()))
    output_data_exp_total = output_data_exp.numpy() if len(output_data_exp_total) == 0 else np.concatenate((output_data_exp_total, output_data_exp.numpy()))
    output_data_exp_pred_total = output_data_exp_pred.numpy() if len(output_data_exp_pred_total) == 0 else np.concatenate((output_data_exp_pred_total, output_data_exp_pred.numpy()))
    output_data_norm_error_total = output_data_norm_error.numpy() if len(output_data_norm_error_total) == 0 else np.concatenate((output_data_norm_error_total, output_data_norm_error.numpy()))


  #     plots
  unique_values = np.unique(input_data_total[:, 1])
  for i in range(len(data_prepocessed.output)):
    out_idx = data_prepocessed.output[i]
    plt.clf()

    #
    y_exp_pred = output_data_exp_pred_total[:,i]
    x_exp = input_data_exp_total[:,0]
    y_exp = output_data_exp_total[:,i]
    x = input_data_total[:,0]
    y = output_data_total[:,i]
    y_error = output_data_norm_error_total[:,i]
    fig, ax = plt.subplots(figsize=(7, 5))
    for value in unique_values:
      # Select data points with the current value in input_data_total[:,1]
      mask = (input_data_total[:,1] == value)
      mask2 = (input_data_exp_total[:,1] == value)

      # Extract data for the current value
      x_ = x[mask]
      y_ = y[mask]
      y_error_ = y_error[mask]
      x_exp_ = x_exp[mask2]
      y_exp_ = y_exp[mask2]
      y_exp_pred_ = y_exp_pred[mask2]


      # Plot with the desired color based on the current value
      if value == 0.01:
          color = 'blue'
          label = "10 mA"
      elif value == 0.02:
          color = 'red'
          label = "20 mA"
      elif value == 0.03:
          color = 'green'
          label = "30 mA"
      elif value == 0.04:
          color = 'purple'
          label = "40 mA"
      elif value == 0.05:
          color = 'grey'
          label = "50 mA"
      else:
          color = 'black'  # Set a default color for other values
          label = "error"

      rse = np.abs((y_exp_pred_ - y_exp_) / y_exp_)
      rse[np.isinf(rse)] = 100  
      # Maximum and Mean RSE
      max_rse = np.max(rse).item()
      mean_rse = np.mean(rse).item()

      # Plot 1: Scatter plot comparing ne_model and calculated ne      
      ax.errorbar(x_/133.322368, y_, xerr=y_error_, yerr=None, fmt='ro', color=color, ecolor='lightblue', capsize=4,  markersize=0.5)
      ax.plot([], [], '-', color=color, markersize=5, label=f'{label}, model')  # Empty plot to add custom legend label
      ax.plot(x_exp_/ 133.322368, y_exp_, 'x',  label=f'{label}, Simulation', markersize = 6, zorder=100, markeredgecolor=color, markerfacecolor='none', linewidth=1)#, zorder=10, markeredgecolor=color, markerfacecolor='none', linewidth=1) #pontos experimentais
    
    plt.style.use(['science','nature'])
    # Set font to sans-serif
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica']
    # Tick params
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=15, width=2)
    ax.tick_params(axis='both', which='minor', labelsize=15, width=2)
    # Ensure all ticks on the x- and y- axes are labeled
    ax.xaxis.set_tick_params(which='both', direction='in', top=True, bottom=True)
    ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)
    # Adjust the font size of the order of magnitude in the plot
    ax.yaxis.get_offset_text().set_fontsize(16)

    # Bold bounding box
    for spine in ax.spines.values():
      spine.set_visible(True)
      spine.set_linewidth(2.5)

    ax.set_xlabel("p (Torr)", fontsize=14, fontweight='bold')
    ax.set_ylabel(out_idx, fontsize=14, fontweight='bold')
    plt.tight_layout()    
    save_path = path + out_idx + '.png'
    plt.savefig(save_path, dpi=300)  # Save with high resolution
    plt.show()

  return output_data_norm, output_data, output_data_exp_norm, output_data_exp, input_data_norm, input_data

