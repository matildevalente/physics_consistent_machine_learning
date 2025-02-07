import os
import torch
import numpy as np
import casadi as ca
import pandas as pd
from contextlib import contextmanager, redirect_stdout

from src.spring_mass_system.utils import set_seed, compute_total_energy

set_seed(42)

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull):
            yield


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
