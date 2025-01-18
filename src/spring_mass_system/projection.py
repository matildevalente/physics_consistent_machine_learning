import os
import torch
import numpy as np
import casadi as ca
import pandas as pd
from scipy.stats import shapiro
from contextlib import contextmanager, redirect_stdout
from typing import Dict, Any
from scipy.optimize import minimize

from src.spring_mass_system.utils import set_seed, compute_total_energy, get_target_trajectory, get_predicted_trajectory
from src.spring_mass_system.plotter.Extra_Figure_1 import plot_output_gaussians



set_seed(42)

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            yield



def perform_shapiro_test(config, all_differences, label):

    # Compute the size of the training set
    N = all_differences.shape[0]
    print("N = ", N)

    # For the Shapiro test to work the dataset size must be < 5000
    if(N > 5000):
        caped_size = 1000
        all_differences_caped = all_differences[:caped_size, :]
    else:
        all_differences_caped = all_differences
    
    x1_diff = np.array(all_differences_caped[:,0])
    v1_diff = np.array(all_differences_caped[:,1])
    x2_diff = np.array(all_differences_caped[:,2])
    v2_diff = np.array(all_differences_caped[:,3])

    # Apply Shapiro-Wilk test
    stat_x1, p_x1 = shapiro(x1_diff)
    stat_v1, p_v1 = shapiro(v1_diff)
    stat_x2, p_x2 = shapiro(x2_diff)
    stat_v2, p_v2 = shapiro(v2_diff)

    # Compute the mean value for the differences of each variable
    diff_x1_mean = np.mean(x1_diff)
    diff_v1_mean = np.mean(v1_diff)
    diff_x2_mean = np.mean(x2_diff)
    diff_v2_mean = np.mean(v2_diff)

    # Print results
    print(label)
    print(f'x1_diff: Statistics={stat_x1:.2f}, p-value={p_x1:.2e}, mean={diff_x1_mean:.2e}')
    print(f'v1_diff: Statistics={stat_v1:.2f}, p-value={p_v1:.2e}, mean={diff_v1_mean:.2e}')
    print(f'x2_diff: Statistics={stat_x2:.2f}, p-value={p_x2:.2e}, mean={diff_x2_mean:.2e}')
    print(f'v2_diff: Statistics={stat_v2:.2f}, p-value={p_v2:.2e}, mean={diff_v2_mean:.2e}')
    print(len(x1_diff))
    plot_output_gaussians(config, x1_diff, v1_diff, x2_diff, v2_diff, label)

def compute_covariance(all_differences):
    N = all_differences.shape[0]

    # Compute covariances: Cov(x,y) = SUM { (x - x_mean) (y - y_mean) } / (N-1) 
    covariance_mtx = np.zeros((4, 4))

    x1_diff = np.array(all_differences[:,0])
    v1_diff = np.array(all_differences[:,1])
    x2_diff = np.array(all_differences[:,2])
    v2_diff = np.array(all_differences[:,3])

    # Compute the mean value for the differences of each variable
    diff_x1_mean = np.mean(x1_diff)
    diff_v1_mean = np.mean(v1_diff)
    diff_x2_mean = np.mean(x2_diff)
    diff_v2_mean = np.mean(v2_diff)

    # Compute the covariances manually
    for i, (var_i, mean_i) in enumerate(zip([0, 1, 2, 3], [diff_x1_mean, diff_v1_mean, diff_x2_mean, diff_v2_mean])):
        for j, (var_j, mean_j) in enumerate(zip([0, 1, 2, 3], [diff_x1_mean, diff_v1_mean, diff_x2_mean, diff_v2_mean])):
            covariance_mtx[i, j] = np.sum((all_differences[:, var_i] - mean_i) * (all_differences[:, var_j] - mean_j)) / (N - 1)

    return torch.tensor(covariance_mtx)

# Compute inverse of covariance matrix on the train set
def get_inverse_covariance_matrix(config, model_nn, device, preprocessed_data):

    ########################## COMPUTE THE COVARIANCE MATRIX BASED ON THE TRAIN LOADER
    model_nn.eval()  # Set the model to evaluation mode
    differences = []
    train_loader = preprocessed_data['train_loader']
    
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model_nn(inputs)

            diff = outputs - targets  # This is (f - p)
            differences.append(diff)
    
    # Concatenate all differences
    all_differences_train_loader = np.array(torch.cat(differences, dim=0))
    perform_shapiro_test(config, all_differences_train_loader, "train_loader")
    covariance_mtx_train_loader = compute_covariance(all_differences_train_loader)
    inverse_covariance_mtx_train_loader = torch.inverse(covariance_mtx_train_loader)  


    ########################## COMPUTE THE COVARIANCE MATRIX BASED ON A TEST TRAJECTORY 
    input_state = inputs[2].clone().detach()
    df_target = get_target_trajectory(config, n_time_steps = 300, initial_state = input_state)
    df_nn   = get_predicted_trajectory(config, preprocessed_data, model_nn,   n_time_steps = 300, initial_state = input_state)
    x1_diff = np.array(df_nn['x1'] - df_target['x1'])
    v1_diff = np.array(df_nn['v1'] - df_target['v1'])
    x2_diff = np.array(df_nn['x2'] - df_target['x2'])
    v2_diff = np.array(df_nn['v2'] - df_target['v2'])
    all_differences_trajectory = np.column_stack((x1_diff, v1_diff, x2_diff, v2_diff))
    perform_shapiro_test(config, all_differences_trajectory, "test trajectory")
    covariance_mtx_test_traj = compute_covariance(all_differences_trajectory)
    inverse_covariance_mtx_test_traj = torch.inverse(covariance_mtx_test_traj)  

    
    return inverse_covariance_mtx_train_loader, inverse_covariance_mtx_test_traj








def revert_normalization(scaler, norm_data):

  not_norm_data = norm_data

  # 2. Revert Min-Max Scaling
  scaler_max = scaler.data_max_
  scaler_min = scaler.data_min_
  not_norm_data = (norm_data + 1) / 2 * (scaler_max - scaler_min) + scaler_min
  
  return not_norm_data

def apply_normalization(scaler, not_norm_data):
  
  # Get min and max used for normalization
  scaler_min  = scaler.data_min_
  scaler_max  = scaler.data_max_

  # apply min max scaling
  feature_norm = (not_norm_data - scaler_min) / (scaler_max - scaler_min) * 2 - 1

  return feature_norm

def constraint_function( out_tensor_norm, initial_condition_norm, scaler_X, config):
  
  # Revert normalization to compute energy
  initial_condition = revert_normalization(scaler_X, initial_condition_norm)
  out_tensor = revert_normalization(scaler_X, out_tensor_norm)
  
  # Compute the Energy of the Initial State of the System
  E_init = compute_total_energy(config, [initial_condition[0], initial_condition[1], initial_condition[2], initial_condition[3]])

  # Create a CasADi SX object using the numerical value
  E_init_sx_value = ca.SX(E_init.item())

  # Compute the Energy of the Next State of the System
  E_out = compute_total_energy(config, [out_tensor[0], out_tensor[1], out_tensor[2], out_tensor[3]])

  # Compute energy residual
  residual = E_out - E_init_sx_value

  return residual

def projection_output(config, y_pred_norm, w_matrix, initial_condition_norm, preprocessed_data):

    # Number of variables (dimension of output)
    n = len(y_pred_norm)

    # Define the symbolic variable - p is the variable to be optimized
    p = ca.SX.sym('x', n)

    # Convert PyTorch tensor to CasADi DM object using the NumPy array
    p0 = ca.DM(np.array(y_pred_norm))              # Define network output (p0)
    W_ca = ca.DM(np.array(w_matrix))                      # Define weight matrix (W_ca)
    scaler_X = preprocessed_data['scaler_X']
    
    # Define the objective function f
    objective = (p - p0).T @ W_ca @ (p - p0)       
    g = constraint_function(p, initial_condition_norm, scaler_X, config)  # Define Constraint Function g

    nlp = {'x': p, 'f': objective, 'g': g}       # Create an NLP solver

    options = {'ipopt.print_level' : 0, 'ipopt.sb' : "no", 'ipopt.tol' : 1e-3, 'ipopt.max_iter' : 100, 'ipopt.acceptable_tol' : 1e-3, 'ipopt.derivative_test' : 'first-order'}

    # Suppress output during solver execution
    with suppress_output():
        solver = ca.nlpsol('solver', 'ipopt', nlp, options)   # Define solver
        # Define the constraint bounds (for equality constraints, the upper and lower bounds are equal)
        lbg = [0] * g.shape[0]  # Zero vector for the lower bound
        ubg = [0] * g.shape[0]  # Zero vector for the upper bound
        sol = solver(x0=p0, lbg=lbg, ubg=ubg)    # Solve the problem

    # extract the normalized projected point
    p_opt_norm = sol['x'].toarray().flatten()                        

    return np.array(p_opt_norm)



def get_projection_df(initial_state, N_test, model, w_matrix, preprocessed_data, config, df_nn):
    # necessary for inverting the normalization
    scaler_X = preprocessed_data['scaler_X']

    # store the predicted of states along the trajectory
    projected_points, energies = [], []

    # initialize the current state with the initial conditions
    current_state = initial_state

    # normalize the current state of the system to allow NN predictions
    initial_state = torch.tensor(np.array(initial_state), dtype=torch.float32) 
    initial_state_norm = torch.tensor(scaler_X.transform((initial_state).reshape(1,-1)).flatten(), dtype = torch.float32)
    projected_points.append(initial_state)

    # compute energy of the initial condition and append
    energy = compute_total_energy(config, [initial_state[0], initial_state[1], initial_state[2], initial_state[3]])
    energies.append(energy.item())
    
    # project each prediction point by point
    for _ in range(N_test-1):

        # 1. normalize the current state of the system to allow NN predictions
        current_state = torch.tensor(np.array(current_state), dtype=torch.float32)
        current_state_norm = torch.tensor(scaler_X.transform((current_state).reshape(1,-1)).flatten(), dtype = torch.float32)

        with torch.no_grad():  

            # 2. predict the next state of the system using the NN / PINN model (normalized)
            next_state_norm = model(current_state_norm)

            # 3. perform point projection onto the manifold
            next_state_norm_proj = projection_output(config, next_state_norm, w_matrix, initial_state_norm, preprocessed_data)
            
            # 4. revert the normalization of the projected point and append the value
            next_state_proj = np.array(torch.tensor(scaler_X.inverse_transform((next_state_norm_proj.reshape(1,-1)))).flatten())
            projected_points.append(next_state_proj)   
        
        # 5. compute energy of projected point
        energy = compute_total_energy(config, [next_state_proj[0], next_state_proj[1], next_state_proj[2], next_state_proj[3]])
        energies.append(energy)

        # 6. update current state 
        current_state = next_state_proj
    
    # Create dataframe
    projected_points = np.stack(projected_points)
    # add predicted states
    df_proj = pd.DataFrame(projected_points, columns=['x1', 'v1', 'x2', 'v2'])
    # add energy of the predicted states
    df_proj['E'] = energies
    # add time for each state 
    df_proj['time(s)'] = df_nn['time(s)']
    
    return df_proj




