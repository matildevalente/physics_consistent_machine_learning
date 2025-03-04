import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.spring_mass_system.utils import set_seed


def preprocess_data(df, print_messages=True):

    set_seed(42)

    # Preprocess data: drop the last row if it contains NaN values
    df = df.dropna()

    # Input features and target labels
    X = df[['x1', 'v1', 'x2', 'v2']].values
    Y = df[['x1(t+dt)', 'v1(t+dt)', 'x2(t+dt)', 'v2(t+dt)']].values

    if(print_messages):
      print("Fitting preprocessing scalers on large dataset...")

    # Normalize input features to range [-1, 1]
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler_X.fit_transform(X)

    # Normalize target labels to range [-1, 1]
    Y_scaled = scaler_X.fit_transform(Y)

    if(print_messages):
        print("Performing train-test-val split...")

    # Train-test-val split (80-10-10)
    # 1. Split: Train (80%) and Temp (20%)
    X_train_norm, X_temp_norm, y_train_norm, y_temp_norm = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42, shuffle=False)
    # 2. Split: Val (50%) and Test (50%) from Temp
    X_val_norm, X_test_norm, y_val_norm, y_test_norm = train_test_split(X_temp_norm, y_temp_norm, test_size=0.5, random_state=42, shuffle=False)

    # Convert arrays to PyTorch tensors
    X_train_norm = torch.tensor(X_train_norm, dtype=torch.float32)
    X_val_norm = torch.tensor(X_val_norm, dtype=torch.float32)
    X_test_norm = torch.tensor(X_test_norm, dtype=torch.float32)
    y_train_norm = torch.tensor(y_train_norm, dtype=torch.float32)
    y_val_norm = torch.tensor(y_val_norm, dtype=torch.float32)
    y_test_norm = torch.tensor(y_test_norm, dtype=torch.float32)

    # Creating TensorDataset for training, validation, and testing
    train_dataset_norm = TensorDataset(X_train_norm, y_train_norm)
    val_dataset_norm = TensorDataset(X_val_norm, y_val_norm)
    test_dataset_norm = TensorDataset(X_test_norm, y_test_norm)

    # Create DataLoader for training, validation, and testing data
    train_loader = DataLoader(train_dataset_norm, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset_norm, batch_size=32, shuffle=False)  # Typically no need to shuffle validation data
    test_loader = DataLoader(test_dataset_norm, batch_size=32, shuffle=False)  # Typically no need to shuffle test data
    
    # Extract the tensors from the train_dataset_norm
    X_train_np = X_train_norm.numpy()
    y_train_np = y_train_norm.numpy()

    # Combine the input and target arrays
    train_data = np.concatenate((X_train_np, y_train_np), axis=1)

    # Define the column names
    columns = ['x1', 'v1', 'x2', 'v2', 'x1_RK', 'v1_RK', 'x2_RK', 'v2_RK']

    # Convert to DataFrame
    train_df = pd.DataFrame(train_data, columns=columns)

    if(print_messages):
        print(f"   → Train ({len(train_dataset_norm)} points)")
        print(f"   → Test  ({len(test_dataset_norm)} points)")
        print(f"   → Val   ({len(val_dataset_norm)} points)")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_df': train_df,
        'scaler_X': scaler_X
    }

