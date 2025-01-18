import os
import sys
import math
import time
import torch 
import random
import numpy as np
import casadi as ca
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from io import StringIO
from itertools import cycle
import torch.optim as optim
device = torch.device("cpu")
from functools import partial
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import linregress
from matplotlib.cm import get_cmap
from scipy.optimize import minimize
from contextlib import contextmanager
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import shapiro



from src.ltp_system.data_prep import LoadDataset
from src.ltp_system.pinn_nn import get_average_predictions
from src.ltp_system.plotter.normality import plot_output_gaussians


# compute the 
def compute_residual(inputs_norm, predictions_norm, model_type, data_prep, error_type):
  
  # 1. Revert Normalization
  inputs_norm_, predictions_norm_ = inputs_norm.clone(), predictions_norm.clone()
  inputs, predictions = inputs_norm_.clone(), predictions_norm_.clone()
  for idx, row in enumerate(predictions_norm):
    inputs[idx, :], pred_denorm = revert_normalization(data_prep, inputs_norm_[idx, :], predictions_norm_[idx, :])
    predictions[idx, :] = torch.cat(pred_denorm)

  # 2. Get Pressure, Current and ne From Model Input
  P_model = inputs[:,0]
  I_model = inputs[:,1]
  ne_model = predictions[:,16]
  
  # 3. Compute Pressure using model's outputs
  num_species = 11
  Tg = predictions[:,11]
  k_b = 1.380649E-23
  concentrations = 0
  for species_idx in range(num_species):
    concentrations += predictions[:,species_idx]
  P_calc = concentrations * Tg * k_b

  # 6. Compute Current using model's outputs
  R = inputs[:,2]
  ne = predictions[:,16]
  vd = predictions[:,14]
  e = 1.602176634e-19
  pi = np.pi
  I_calc = e * ne * vd * pi * R * R

  # 7. Compute ne using model's outputs
  ne_calc = predictions[:,4] + predictions[:,7] - predictions[:,8]         # ne = O2(+,X) + O(+,gnd) - O(-,gnd)

  # 8. Compute errors - in the case of RMSE apply normalization to guarantee the error has no units
  if error_type == 'mape':
    p_error, p_sem = _compute_mape_and_sem(P_calc, P_model)
    i_error, i_sem = _compute_mape_and_sem(I_calc, I_model)
    ne_error, ne_sem = _compute_mape_and_sem(ne_calc, ne_model)

  elif error_type == 'rmse':
    #   5.1. Pressure   
    P_model = apply_normalization(P_model, data_prep.scalers_input[0], 0, data_prep.skewed_features_in)
    P_calc  = apply_normalization(P_calc , data_prep.scalers_input[0], 0, data_prep.skewed_features_in)
    #   5.2. Current 
    I_model = apply_normalization(I_model, data_prep.scalers_input[1], 1, data_prep.skewed_features_in)
    I_calc  = apply_normalization(I_calc , data_prep.scalers_input[1], 1, data_prep.skewed_features_in)
    #   5.3. ne 
    ne_model = apply_normalization(ne_model, data_prep.scalers_output[16], 16, data_prep.skewed_features_out)
    ne_calc  = apply_normalization(ne_calc , data_prep.scalers_output[16], 16, data_prep.skewed_features_out)

    #   5.4. Compute RMSE and SEM
    p_error, p_sem = _compute_rmse_and_sem(P_calc, P_model)
    i_error, i_sem = _compute_rmse_and_sem(I_calc, I_model)
    ne_error, ne_sem = _compute_rmse_and_sem(ne_calc, ne_model)

  # 9. Return results
  results = {
    f'p_{error_type}': p_error,
    f'i_{error_type}': i_error,
    f'ne_{error_type}': ne_error,
    f'p_sem': p_sem,
    f'i_sem': i_sem,
    f'ne_sem': ne_sem, 
  }

  return {model_type: results}

# Compute the MAPE and the SEM of the NN/PINN and of the projections with different constraints applied on the test set from a given .txt file
def compute_rmse_physical_laws(dataset_name, data_preprocessed, models, w_matrix, model_type):
  
  # 1. Load and extract experimental dataset
  test_dataset = LoadDataset(dataset_name)
  test_targets, test_inputs = test_dataset.y, test_dataset.x

  # 2. Apply log transform to the skewed features
  if len(data_preprocessed.skewed_features_in) > 0:
      test_inputs[:, data_preprocessed.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, data_preprocessed.skewed_features_in]))

  if len(data_preprocessed.skewed_features_out) > 0:
      test_targets[:, data_preprocessed.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, data_preprocessed.skewed_features_out]))

  # 3. normalize targets with the model used on the training data
  normalized_inputs = torch.cat([torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) for i, scaler in enumerate(data_preprocessed.scalers_input)], dim=1)

  # 4. generate predictions using the trained models
  normalized_model_predictions =  get_average_predictions(models, torch.tensor(normalized_inputs))
  normalized_proj_predictions  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions), torch.tensor(normalized_inputs), data_preprocessed, constraint_p_i_ne, w_matrix) 
  normalized_proj_predictions = torch.tensor(np.stack(normalized_proj_predictions))

  # 5. compute mape and sem for the compliance with physical laws for NN and its projected predictions
  proj_nn_results = compute_residual(normalized_inputs, normalized_proj_predictions, model_type + "_proj", data_preprocessed, error_type = 'rmse')
  nn_results = compute_residual(normalized_inputs, normalized_model_predictions, model_type, data_preprocessed, error_type = 'rmse')

  return {**nn_results, **proj_nn_results}

# Compute the MAPE and the SEM of the NN/PINN and of the projections with different constraints applied on the test set from a given .txt file
def compute_errors_physical_laws_loki(dataset_name, data_preprocessed, w_matrix, error_type):
  
  # 1. Load and extract experimental dataset
  test_dataset = LoadDataset(dataset_name)
  test_targets, test_inputs = test_dataset.y, test_dataset.x

  # 2. Apply log transform to the skewed features
  if len(data_preprocessed.skewed_features_in) > 0:
      test_inputs[:, data_preprocessed.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, data_preprocessed.skewed_features_in]))

  if len(data_preprocessed.skewed_features_out) > 0:
      test_targets[:, data_preprocessed.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, data_preprocessed.skewed_features_out]))

  # 3. normalize targets with the model used on the training data
  normalized_inputs  = torch.cat([torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) for i, scaler in enumerate(data_preprocessed.scalers_input)], dim=1)
  normalized_targets = torch.cat([torch.from_numpy(scaler.transform(test_targets[:, i:i+1])) for i, scaler in enumerate(data_preprocessed.scalers_output)], dim=1)

  # 4. generate predictions using the trained models
  normalized_proj_targets  =  get_average_predictions_projected(torch.tensor(normalized_targets), torch.tensor(normalized_inputs), data_preprocessed, constraint_p_i_ne, w_matrix) 
  normalized_proj_targets = torch.tensor(np.stack(normalized_proj_targets))

  # 5. compute mape and sem for the compliance with physical laws for NN and its projected predictions
  proj_loki_results = compute_residual(normalized_inputs, normalized_proj_targets, "loki_proj", data_preprocessed, error_type)
  loki_results = compute_residual(normalized_inputs, normalized_targets, "loki", data_preprocessed, error_type)

  return {**loki_results, **proj_loki_results}











# Compute the MAPE and the SEM of the NN/PINN and of the projections with different constraints applied on the test set from a given .txt file
def compute_mape_physical_laws(dataset_name, data_preprocessed, models, w_matrix, model_type):
  
  # 1. Load and extract experimental dataset
  test_dataset = LoadDataset(dataset_name)
  test_targets, test_inputs = test_dataset.y, test_dataset.x

  # 2. Apply log transform to the skewed features
  if len(data_preprocessed.skewed_features_in) > 0:
      test_inputs[:, data_preprocessed.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, data_preprocessed.skewed_features_in]))

  if len(data_preprocessed.skewed_features_out) > 0:
      test_targets[:, data_preprocessed.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, data_preprocessed.skewed_features_out]))

  # 3. normalize targets with the model used on the training data
  normalized_inputs = torch.cat([torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) for i, scaler in enumerate(data_preprocessed.scalers_input)], dim=1)

  # 4. generate predictions using the trained models
  normalized_model_predictions =  get_average_predictions(models, torch.tensor(normalized_inputs))
  normalized_proj_predictions  =  get_average_predictions_projected(torch.tensor(normalized_model_predictions), torch.tensor(normalized_inputs), data_preprocessed, constraint_p_i_ne, w_matrix) 
  normalized_proj_predictions = torch.tensor(np.stack(normalized_proj_predictions))

  # 5. compute mape and sem for the compliance with physical laws for NN and its projected predictions
  proj_nn_results = compute_residual(normalized_inputs, normalized_proj_predictions, model_type + "_proj", data_preprocessed, error_type = 'mape')
  nn_results = compute_residual(normalized_inputs, normalized_model_predictions, model_type, data_preprocessed, error_type = 'mape')

  return {**nn_results, **proj_nn_results}

# Compute the MAPE and the SEM of the NN/PINN and of the projections with different constraints applied on the test set from a given .txt file
def compute_projection_results(config_model, w_matrix, dataset_name, data_preprocessed, models, error_type):

    # 1. Determine the model type being analyzed
    model_type = "NN" if all(param == 0 for param in config_model['lambda_physics']) else "PINN"

    # 2. Load and extract experimental dataset
    test_dataset = LoadDataset(dataset_name)
    test_targets, test_inputs = test_dataset.y, test_dataset.x

    # 3. Apply log transform to the skewed features
    if len(data_preprocessed.skewed_features_in) > 0:
        test_inputs[:, data_preprocessed.skewed_features_in] = torch.log1p(torch.tensor(test_inputs[:, data_preprocessed.skewed_features_in]))

    if len(data_preprocessed.skewed_features_out) > 0:
        test_targets[:, data_preprocessed.skewed_features_out] = torch.log1p(torch.tensor(test_targets[:, data_preprocessed.skewed_features_out]))

    # 4. normalize targets with the model used on the training data
    normalized_inputs = torch.cat([
        torch.from_numpy(scaler.transform(test_inputs[:, i:i+1])) 
        for i, scaler in enumerate(data_preprocessed.scalers_input)
        ], dim=1
    )

    normalized_test_targets = torch.cat([
        torch.from_numpy(scaler.transform(test_targets[:, i:i+1])) 
        for i, scaler in enumerate(data_preprocessed.scalers_output)
        ], dim=1
    )
    
    _, denormalized_test_targets = data_preprocessed.inverse_transform(normalized_inputs, normalized_test_targets)
    
    # 5. Define constraints and their combinations
    constraint_combinations = { 
      "model": [model_type], 
      "P": constraint_p, #["P"], 
      "I": constraint_i, #["I"], 
      "ne": constraint_ne, #["ne"],
      "P_I": constraint_p_i, #["P", "I"], 
      "P_ne": constraint_p_ne, #["P", "ne"], 
      "I_ne": constraint_i_ne, #["I", "ne"], 
      "P_I_ne": constraint_p_i_ne #["P", "I", "ne"]
    }
    
    # 6. Initialize the results dictionary
    results = {key: {"constraints": value, error_type: [], "sem": []} for key, value in constraint_combinations.items()}

    # 7. Generate predictions using the trained models
    normalized_model_predictions =  get_average_predictions(models, torch.tensor(normalized_inputs))
    _, denormalized_model_predictions = data_preprocessed.inverse_transform(normalized_inputs, normalized_model_predictions)
    # Calculate ERROR and SEM here
    for i in range(len(data_preprocessed.output_features)):
        if error_type == 'mape':
            error_value, sem_value = _compute_mape_and_sem(denormalized_test_targets[:, i], denormalized_model_predictions[:, i])
        elif error_type == 'rmse':
            error_value, sem_value = _compute_rmse_and_sem(normalized_test_targets[:, i], normalized_model_predictions[:, i])
        results["model"][error_type].append(error_value)
        results["model"]["sem"].append(sem_value)

    # 8. Generate predictions for each constraint combination and compute error metrics
    for constraint_key, constraints in constraint_combinations.items():

        # Skip the "model" key as it's already handled
        if constraint_key == "model":
            continue  

        # Apply projection to model predictions
        normalized_projected_predictions = get_average_predictions_projected(torch.tensor(normalized_model_predictions), torch.tensor(normalized_inputs), data_preprocessed, constraints, w_matrix) 

        # Inverse transform the predictions to original scale
        _, denormalized_projected_predictions = data_preprocessed.inverse_transform(normalized_inputs, normalized_projected_predictions)

        # Convert to torch tesor
        normalized_projected_predictions = torch.tensor(np.stack(normalized_projected_predictions))

        # Compute error metrics for each output variable
        for i in range(len(data_preprocessed.output_features)):
    
            # Calculate ERROR and SEM 
            if error_type == 'mape':
                actual_values = denormalized_test_targets[:, i]
                predicted_values = denormalized_projected_predictions[:, i]
                error_value, sem_value = _compute_mape_and_sem(actual_values, predicted_values)
            elif error_type == 'rmse':
                actual_values = normalized_test_targets[:, i]
                predicted_values = normalized_projected_predictions[:, i]
                error_value, sem_value = _compute_rmse_and_sem(actual_values, predicted_values)

            results[constraint_key][error_type].append(error_value)
            results[constraint_key]["sem"].append(sem_value)
    
    return results

# Compute MAPE and SEM given the targets and the predictions
def _compute_mape_and_sem(targets, predictions):
  # Compute MAPE
  mape_values = np.abs((predictions - targets)) / np.abs(targets) * 100
  mape_arr = np.array(mape_values)
  mape_value = np.mean(mape_arr).item()
  
  # Compute SEM
  n_values = len(mape_arr)
  sem_value = np.std(mape_arr) / np.sqrt(n_values)
  
  return mape_value, sem_value

# Compute RMSE and SEM given the targets and the predictions
def _compute_rmse_and_sem(targets, predictions):
  # Convert to numpy arrays
  predictions_arr = np.array(predictions)
  targets_arr = np.array(targets)

  # Compute RMSE
  # Compute squared errors for each prediction
  squared_errors = (predictions_arr - targets_arr) ** 2
  
  # Compute RMSE for each prediction
  rmse_values = np.sqrt(squared_errors)
  
  # Take mean of RMSE values
  rmse_value = np.mean(rmse_values)
  
  # Compute SEM
  sem_value = np.std(rmse_values) / np.sqrt(len(rmse_values))
  
  return rmse_value, sem_value



####################################################################################################
################################## PHYSICAL CONSTRAINTS FUNCTIONS ##################################
####################################################################################################
# --------------------------------------------------------------- Residual computation
# Define discharge current physical constraints
def get_current_law_residual(x, p, data_preprocessed):
  
  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 2. Get Current From Model Input
  I_model = x[1]  

  # 3. Compute Current using physical law
  R = x[2]
  ne = p[16]
  vd = p[14]
  e = 1.602176634e-19
  pi = np.pi
  I_calc = e * ne * vd * pi * R * R

  # 4. Apply Normalization - Current 
  I_model = apply_normalization(I_model, data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)
  I_calc  = apply_normalization(I_calc , data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)

  return I_model,  I_calc

def get_pressure_law_residual(x, p, data_preprocessed):

  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 2. Get Pressure From Model Input
  P_model = x[0]
  
  # 3. Compute pressure using physical law
  num_species = 11
  Tg = p[11]
  k_b = 1.380649E-23
  concentrations = 0
  for species_idx in range(num_species):
    concentrations += p[species_idx]
  P_calc = concentrations * Tg * k_b
  
  # 4. Apply Normalization - Pressure   
  P_model = apply_normalization(P_model, data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  P_calc  = apply_normalization(P_calc , data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)

  return P_model, P_calc

def get_quasi_neutrality_law_residual(x, p, data_preprocessed):

  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 2. Get Pressure From Model Output
  ne_model = p[16]  

  # 3. Compute ne using physical law
  ne_calc = p[4] + p[7] - p[8]       # ne = O2(+,X) + O(+,gnd) - O(-,gnd)

  # 4. Apply Normalization - ne 
  ne_model = apply_normalization(ne_model, data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)
  ne_calc  = apply_normalization(ne_calc , data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)

  return ne_model, ne_calc

# --------------------------------------------------------------- Constraint functions
# Define discharge current physical constraints
def constraint_i(x, p, data_preprocessed):

  I_model,  I_calc = get_current_law_residual(x, p, data_preprocessed)

  return ca.vertcat(I_model - I_calc)

# Define pressure physical constraints
def constraint_p(x, p, data_preprocessed):
    
  P_model, P_calc = get_pressure_law_residual(x, p, data_preprocessed)

  # 4. Return Result
  return ca.vertcat(P_model - P_calc)

# Define quasi-neutrality physical constraints
def constraint_ne(x, p, data_preprocessed):
  
  ne_model, ne_calc = get_quasi_neutrality_law_residual(x, p, data_preprocessed)
    
  return ca.vertcat(ne_model - ne_calc)

# Physical constraint function that contains all 3 constraints
def constraint_p_i_ne(x, p, data_preprocessed):
  
  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 1. Get Pressure, Current and ne From Model Input
  P_model = x[0]
  I_model = x[1]
  ne_model = p[16]

  # 5. Compute Pressure using model's outputs
  num_species = 11
  Tg = p[11]
  k_b = 1.380649E-23
  concentrations = 0
  for species_idx in range(num_species):
    concentrations += p[species_idx]
  P_calc = concentrations * Tg * k_b

  # 6. Compute Current using model's outputs
  R = x[2]
  ne = p[16]
  vd = p[14]
  e = 1.602176634e-19
  pi = np.pi
  I_calc = e * ne * vd * pi * R * R

  # 7. Compute ne using model's outputs
  ne_calc = p[4] + p[7] - p[8]         # ne = O2(+,X) + O(+,gnd) - O(-,gnd)

  # 5. Apply Normalization
  #   5.1. Pressure   
  P_model = apply_normalization(P_model, data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  P_calc  = apply_normalization(P_calc , data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  #   5.2. Current 
  I_model = apply_normalization(I_model, data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)
  I_calc  = apply_normalization(I_calc , data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)
  #   5.3. ne 
  ne_model = apply_normalization(ne_model, data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)
  ne_calc  = apply_normalization(ne_calc , data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)

  # 6. Define constraint vector
  g_x_p = ca.vertcat(P_model - P_calc, I_model - I_calc, ne_model - ne_calc)


  return g_x_p

# Physical constraint function that contains 2 constraints - pressure and current
def constraint_p_i(x, p, data_preprocessed):
  
  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 1. Get Pressure, Current and ne From Model Input
  P_model = x[0]
  I_model = x[1]

  # 5. Compute Pressure using model's outputs
  num_species = 11
  Tg = p[11]
  k_b = 1.380649E-23
  concentrations = 0
  for species_idx in range(num_species):
    concentrations += p[species_idx]
  P_calc = concentrations * Tg * k_b

  # 6. Compute Current using model's outputs
  R = x[2]
  ne = p[16]
  vd = p[14]
  e = 1.602176634e-19
  pi = np.pi
  I_calc = e * ne * vd * pi * R * R


  # 5. Apply Normalization
  #   5.1. Pressure   
  P_model = apply_normalization(P_model, data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  P_calc  = apply_normalization(P_calc , data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  #   5.2. Current 
  I_model = apply_normalization(I_model, data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)
  I_calc  = apply_normalization(I_calc , data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)

  # 6. Define constraint vector
  g_x_p = ca.vertcat(P_model - P_calc, I_model - I_calc)

  # 7. Compute the L2 norm of the vector
  #norm_g_x_p = ca.norm_2(g_x_p)

  return g_x_p

# Physical constraint function that contains 2 constraints - pressure and quasi-neutrality
def constraint_p_ne(x, p, data_preprocessed):
  
  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 1. Get Pressure, Current and ne From Model Input
  P_model = x[0]
  ne_model = p[16]

  # 5. Compute Pressure using model's outputs
  num_species = 11
  Tg = p[11]
  k_b = 1.380649E-23
  concentrations = 0
  for species_idx in range(num_species):
    concentrations += p[species_idx]
  P_calc = concentrations * Tg * k_b

  # 7. Compute ne using model's outputs
  ne_calc = p[4] + p[7] - p[8]         # ne = O2(+,X) + O(+,gnd) - O(-,gnd)

  # 5. Apply Normalization
  #   5.1. Pressure   
  P_model = apply_normalization(P_model, data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  P_calc  = apply_normalization(P_calc , data_preprocessed.scalers_input[0], 0, data_preprocessed.skewed_features_in)
  #   5.3. ne 
  ne_model = apply_normalization(ne_model, data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)
  ne_calc  = apply_normalization(ne_calc , data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)

  # 6. Define constraint vector
  g_x_p = ca.vertcat(P_model - P_calc, ne_model - ne_calc)

  # 7. Compute the L2 norm of the vector
  #norm_g_x_p = ca.norm_2(g_x_p)

  return g_x_p

# Physical constraint function that contains 2 constraints - current and quasi-neutrality
def constraint_i_ne(x, p, data_preprocessed):
  
  # 1. Revert Normalization
  x, p = revert_normalization(data_preprocessed, x, p)

  # 1. Get Pressure, Current and ne From Model Input
  I_model = x[1]
  ne_model = p[16]

  # 6. Compute Current using model's outputs
  R = x[2]
  ne = p[16]
  vd = p[14]
  e = 1.602176634e-19
  pi = np.pi
  I_calc = e * ne * vd * pi * R * R

  # 7. Compute ne using model's outputs
  ne_calc = p[4] + p[7] - p[8]         # ne = O2(+,X) + O(+,gnd) - O(-,gnd)

  # 5. Apply Normalization
  #   5.2. Current 
  I_model = apply_normalization(I_model, data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)
  I_calc  = apply_normalization(I_calc , data_preprocessed.scalers_input[1], 1, data_preprocessed.skewed_features_in)
  #   5.3. ne 
  ne_model = apply_normalization(ne_model, data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)
  ne_calc  = apply_normalization(ne_calc , data_preprocessed.scalers_output[16], 16, data_preprocessed.skewed_features_out)

  # 6. Define constraint vector
  g_x_p = ca.vertcat(I_model - I_calc, ne_model - ne_calc)

  # 7. Compute the L2 norm of the vector
  #norm_g_x_p = ca.norm_2(g_x_p)

  return g_x_p


####################################################################################################
################################## NORMALIZATION FUNCTIONS           ###############################
####################################################################################################

# --------------------------------------------------------------- Apply and revert normalization with casadi objects
# Function to revert normalization of outputs and inputs
def revert_normalization(data_preprocessed, norm_data_in, norm_data_out):

  not_norm_data_out = [ca.SX() for _ in range(len(data_preprocessed.scalers_output))]
  not_norm_data_in = norm_data_in

  # 2. Revert Min-Max Scaling
  # 2.1. Revert Inputs
  for in_idx in range(len(data_preprocessed.scalers_input)):
    in_scaler_max = data_preprocessed.scalers_input[in_idx].data_max_
    in_scaler_min = data_preprocessed.scalers_input[in_idx].data_min_
    not_norm_data_in[in_idx] = (norm_data_in[in_idx] + 1) / 2 * (in_scaler_max - in_scaler_min) + in_scaler_min
  # 2.2. Revert Outputs
  for out_idx in range(len(data_preprocessed.scalers_output)):
    out_scaler_max = data_preprocessed.scalers_output[out_idx].data_max_
    out_scaler_min = data_preprocessed.scalers_output[out_idx].data_min_
    not_norm_data_out[out_idx] = (norm_data_out[out_idx] + 1) / 2 * (out_scaler_max - out_scaler_min) + out_scaler_min

  # 3. Revert Log Transformation for skewed features
  # 3.1. Revert Inputs
  for in_idx in range(len(data_preprocessed.scalers_input)):
    if in_idx in data_preprocessed.scalers_input:
      not_norm_data_in[in_idx] = ca.exp(not_norm_data_in[in_idx]) - 1
  # 3.2. Revert Outputs
  for out_idx in range(len(data_preprocessed.scalers_output)):
    if out_idx in data_preprocessed.scalers_output:
      not_norm_data_out[out_idx] = ca.exp(not_norm_data_out[out_idx]) - 1

  return not_norm_data_in, not_norm_data_out

# Function to apply normalization to outputs and inputs
def apply_normalization(feature_not_norm, scaler, index_, skewed_features_in_or_out):
  scaler_min  = scaler.data_min_
  scaler_max  = scaler.data_max_

  # apply log transformation
  if index_ in skewed_features_in_or_out:
    feature_not_norm = ca.log(feature_not_norm + 1)

  # apply min max scaling
  feature_norm = (feature_not_norm - scaler_min) / (scaler_max - scaler_min) * 2 - 1

  return feature_norm


####################################################################################################
################################## CASADI OPTIMIZATION ALGORITHM  ##################################
####################################################################################################
@contextmanager
def suppress_output():
    devnull = open(os.devnull, 'w')
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull.close()
  
# Project the output of the NN/PINN model using the W matrix
def project_output(x, y_pred, constraint_function, data_preprocessed, W):

  # Number of variables (dimension of output)
  n = len(y_pred[0])

  # Define the symbolic variable - p is the variable to be optimized
  p = ca.SX.sym('x', n)

  # Convert PyTorch tensor to CasADi DM object using the NumPy array
  x_DM = ca.DM(x.numpy())
  p0 = ca.DM(np.array(y_pred[0]))              # Define network output (p0)
  W_ca = ca.DM(np.array(W))                    # Define weight matrix (W_ca)

  objective = (p - p0).T @ W_ca @ (p - p0)     # Define the objective function f
  g = constraint_function( x_DM, p, data_preprocessed)    # Define Constraint Function g

  nlp = {'x': p, 'f': objective, 'g': g}       # Create an NLP solver

  options = {'ipopt.print_level' : 0, 'ipopt.sb' : "no", 'ipopt.tol' : 1e-8, 'ipopt.max_iter' : 200, 'ipopt.acceptable_tol' : 1e-8, 'ipopt.derivative_test' : 'second-order'}

  # Suppress output during solver execution
  with suppress_output():
    solver = ca.nlpsol('solver', 'ipopt', nlp, options)   # Define solver
    # Define the constraint bounds (for equality constraints, the upper and lower bounds are equal)
    lbg = [0] * g.shape[0]  # Zero vector for the lower bound
    ubg = [0] * g.shape[0]  # Zero vector for the upper bound
    sol = solver(x0=p0, lbg=lbg, ubg=ubg)    # Solve the problem

  p_opt = sol['x'].toarray().flatten()                        # Extract and print the solution and Convert to numpy array 

  #print(solver.stats())


  
  return p_opt

# Method to project the predictions of the NN and PINN models 
def get_average_predictions_projected(y_pred, X_test_norm, data_preprocessed, constraint_func, W):
  pred = []
  with torch.no_grad():
    for input_tensor_x, input_tensor_y in zip(X_test_norm, y_pred):
      
      predictions_projected = project_output(
        input_tensor_x.unsqueeze(0),
        input_tensor_y.unsqueeze(0),
        constraint_func,
        data_preprocessed,
        W
      )
      
      pred.append(predictions_projected)
  
  return pred



####################################################################################################
################################## FUNCTIONS USED IN W MATRIX STUDY  ###############################
####################################################################################################

# Perform Shapiro test on the differences between the model predictions and the test targets
def perform_shapiro_test(config, all_differences):

    # Compute the size of the training set
    N = all_differences.shape[0]
    len_outputs = len(all_differences[0]) 
    variable_names = config['dataset_generation']['output_features']

    print("N = ", N)
    print("len_outputs = ", len_outputs)

    # For the Shapiro test to work the dataset size must be < 5000
    if(N > 5000):
        caped_size = 1000
        all_differences_caped = all_differences[:caped_size, :]
    else:
        all_differences_caped = all_differences
    
    # Extract differences and compute statistics
    diff_arrays = [np.array(all_differences_caped[:, i]) for i in range(len_outputs)]
    shapiro_results = [shapiro(diff) for diff in diff_arrays]
    diff_means = [np.mean(diff) for diff in diff_arrays]

    # Print results
    for i, var_name in enumerate(variable_names):
        stat, p_value = shapiro_results[i]
        mean = diff_means[i]
        print(f'{var_name}_diff: Statistics={stat:.2f}, p-value={p_value:.2e}, mean={mean:.2e}')

    print(len(diff_arrays[0]))
    plot_output_gaussians(config, *diff_arrays)

# Compute the covariance matrix of the differences between the model predictions and the test targets
def compute_covariance(all_differences):
    N = all_differences.shape[0]
    
    # Compute covariances: Cov(x,y) = SUM { (x - x_mean) (y - y_mean) } / (N-1) 
    len_outputs = len(all_differences[0]) 
    covariance_mtx = np.zeros((len_outputs, len_outputs))

    # Compute the mean value for the differences of each variable
    diff_means = np.mean(all_differences, axis=0)

    # Compute the covariances using nested loops
    for i in range(len_outputs):
        for j in range(len_outputs):
            covariance_mtx[i, j] = np.sum((all_differences[:, i] - diff_means[i]) * 
                                          (all_differences[:, j] - diff_means[j])) / (N - 1)

    return torch.tensor(covariance_mtx)

# Compute inverse of covariance matrix on the train set
def get_inverse_covariance_matrix(config, preprocessed_data, nn_models, device_nn): 
  config_model = config['nn_model']
  train_loader = torch.utils.data.DataLoader(preprocessed_data.train_data, batch_size=config_model['batch_size'], shuffle=True)
  differences = []
  
  with torch.no_grad():
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = get_average_predictions(nn_models, inputs)
      diff = outputs - targets  # This is (f - p)
      differences.append(diff)
    
    # Concatenate all differences
    all_differences_train_loader = np.array(torch.cat(differences, dim=0))
    perform_shapiro_test(config, all_differences_train_loader)
    covariance_mtx_train_loader = compute_covariance(all_differences_train_loader)
    inverse_covariance_mtx_train_loader = torch.inverse(covariance_mtx_train_loader)  

    
    return inverse_covariance_mtx_train_loader

# Print a matrix with 3 decimal places for each element and aligned columns
def print_matrix(matrix):
  # Convert tensor to numpy array for easier printing
  matrix_np = matrix.numpy()

  # Print the matrix with 3 decimal places for each element and aligned columns
  for row in matrix_np:
      formatted_row = [f"{item:8.3f}" for item in row]  # Each value will take up 8 characters, aligned by decimal
      print(" ".join(formatted_row))

   

