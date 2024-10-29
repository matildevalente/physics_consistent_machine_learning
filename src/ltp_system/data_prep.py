import torch 
import warnings 
import numpy as np
import pandas as pd
from scipy.stats import skew 
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
device = torch.device("cpu")


# 1. Define the class with the preprocessing methods
class DataPreprocessor():
  # initialize object
  def __init__(self, config):
    self.output_features = config['dataset_generation']['output_features']
    self.input_features  = config['dataset_generation']['input_features']
    
    self.fraction_train = config['data_prep']['fraction_train']
    self.fraction_val   = config['data_prep']['fraction_val']
    self.fraction_test  = config['data_prep']['fraction_test']
    
    self.skew_threshold_down = config['data_prep']['skew_threshold_down']
    self.skew_threshold_up   = config['data_prep']['skew_threshold_up']

    self.skewed_features_in  = None
    self.skewed_features_out = None
    self.scalers_output = None
    self.scalers_input = None

    self.train_data = None
    self.test_data = None
    self.val_data = None

    self.df_train = None
    self.df_test = None
    self.df_val = None

    self.X_val = None
    self.y_val = None
    self.X_val_norm = None
    self.y_val_norm = None

    self.X_train = None
    self.y_train = None
    self.X_train_norm = None
    self.y_train_norm = None

    self.X_test = None
    self.y_test = None
    self.X_test_norm = None
    self.y_test_norm = None

  # train-test-split methods
  def setup_dataset(self, X_data, y_data):

    # DATA PREPARATION -------------------------------------------------------------------------------------------
    # 1. Train-test-validation split
    train_size = int(self.fraction_train * len(X_data)) 
    val_size = int(self.fraction_val * len(X_data)) 
    test_size = len(X_data) - train_size - val_size 
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=4) 
    self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=val_size, random_state=4) 
    
    # 2. Normalization
    self.y_train_norm, self.y_test_norm, self.y_val_norm, self.scalers_output, self.skewed_features_out  = self.normalize_dataset(torch.tensor(self.y_train), torch.tensor(self.y_test), torch.tensor(self.y_val))
    self.X_train_norm, self.X_test_norm, self.X_val_norm, self.scalers_input, self.skewed_features_in  = self.normalize_dataset(torch.tensor(self.X_train), torch.tensor(self.X_test), torch.tensor(self.X_val))

    # 3. Convert Data to Torch Tensors
    self.train_data = Data(self.X_train_norm, self.y_train_norm)
    self.test_data  = Data(self.X_test_norm, self.y_test_norm)
    self.val_data   = Data(self.X_val_norm, self.y_val_norm)

    # 4. Create pandas DataFrames
    self.df_train = pd.DataFrame(np.hstack((self.X_train_norm, self.y_train_norm)), columns = self.input_features + self.output_features)
    self.df_test = pd.DataFrame(np.hstack((self.X_test_norm, self.y_test_norm)), columns = self.input_features + self.output_features)
    self.df_val = pd.DataFrame(np.hstack((self.X_val_norm, self.y_val_norm)), columns = self.input_features + self.output_features)

    return self

  # apply normalization & log-transform
  def normalize_dataset(self, train_data, test_data, val_data):
    
    # 1. Compute squewness of all features
    skewness = np.abs(np.array([skew((train_data).cpu().numpy()[:, i]) for i in range((train_data).shape[1])]))

    # 2. Determine squewed features by comparing to a threshold
    skewed_features = np.where((skewness > self.skew_threshold_up) | (skewness < self.skew_threshold_down))[0]
    print("The skewed features are: " + ", ".join([self.output_features[i] for i in skewed_features.tolist()])) if len(skewed_features) > 0 else None
    print("\n")
    
    # 3. Apply the log transform to the skewed features
    trainDataTransformed, testDataTransformed, valDataTransformed = (train_data).clone(), (test_data).clone(), (val_data).clone()
    if len(skewed_features) > 0:    
      # Apply log transformation to trainData testData and valData
      trainDataTransformed[:, skewed_features] = torch.log1p(trainDataTransformed[:, skewed_features])
      testDataTransformed[:, skewed_features] = torch.log1p(testDataTransformed[:, skewed_features])
      valDataTransformed[:, skewed_features] = torch.log1p(valDataTransformed[:, skewed_features])

    # 4. Fit MinMaxScaler to data
    scalers = []
    for i in range(trainDataTransformed.shape[1]):
        scaler = MinMaxScaler(feature_range=(-1, 1)).fit(trainDataTransformed.cpu().numpy()[:, i:i+1])
        scalers.append(scaler)

    # 5. Apply the min max scaler
    trainDataScaled = torch.cat([torch.tensor(scaler.transform(trainDataTransformed.cpu().numpy()[:, i:i+1])) for i, scaler in enumerate(scalers)], dim=1)
    testDataScaled = torch.cat([torch.tensor(scaler.transform(testDataTransformed.cpu().numpy()[:, i:i+1])) for i, scaler in enumerate(scalers)], dim=1)
    valDataScaled = torch.cat([torch.tensor(scaler.transform(valDataTransformed.cpu().numpy()[:, i:i+1])) for i, scaler in enumerate(scalers)], dim=1)

    return np.array(trainDataScaled), np.array(testDataScaled), np.array(valDataScaled), scalers, skewed_features

  # reverse transformations - min max & log-transform
  def inverse_transform(self, norm_data_in, norm_data_out):
    # 1. Clone
    orginal_data_in  = (torch.tensor(norm_data_in)).clone()
    orginal_data_out = (torch.tensor(norm_data_out)).clone()
    
    # 2. Revert Min-Max Scaling
    for i, scaler in enumerate(self.scalers_input):
      orginal_data_in[:, i:i+1] = torch.tensor(scaler.inverse_transform(orginal_data_in[:, i:i+1].cpu().numpy()))
    for i, scaler in enumerate(self.scalers_output):
      orginal_data_out[:, i:i+1] = torch.tensor(scaler.inverse_transform(orginal_data_out[:, i:i+1].cpu().numpy()))
    
    # 3. Revert Log Transformation for skewed features
    if len(self.skewed_features_in) > 0:
      orginal_data_in[:, self.skewed_features_in] = torch.expm1(orginal_data_in[:, self.skewed_features_in])
    if len(self.skewed_features_out) > 0:
      orginal_data_out[:, self.skewed_features_out] = torch.expm1(orginal_data_out[:, self.skewed_features_out])

    return orginal_data_in, orginal_data_out

  # inverse transformation for dataframes
  def apply_inverse_transform(self, df):
    # 1. Extract Input and Output Cols
    input_columns = df.iloc[:, :len(self.input_features)].values
    output_columns = df.iloc[:, -len(self.output_features):].values
    # 2. Apply Method to Reverse Transformation
    original_input_cols, original_output_cols = self.inverse_transform(input_columns, output_columns)
    df.iloc[:, :len(self.input_features)] = original_input_cols.cpu().numpy().astype(np.float64)
    df.iloc[:, -len(self.output_features):] = original_output_cols.cpu().numpy().astype(np.float64)

    return df

# 2. Class to Extract Data from .txt Files
class LoadDataset(torch.utils.data.Dataset):
  # first 3 columns: inputs
  # last 17 columns: outputs

  def __init__(self, src_file, m_rows=None):
    delimiter=' '
    all_xy = np.genfromtxt(src_file, max_rows=m_rows,
      usecols=range(20), delimiter=delimiter,
      comments="#", dtype=np.float64)
    
    self.datapoints = np.genfromtxt(src_file, max_rows=m_rows, usecols=range(20), delimiter=delimiter, comments="#", dtype=np.float64)
    self.x, self.y = self.datapoints[:, :3], self.datapoints[:, 3:]
    self.len = len(self.x)

    tmp_x, tmp_y = all_xy[:, :3], all_xy[:, 3:]

    self.x_data = torch.tensor(tmp_x, \
      dtype=torch.float64).to(device)
    self.y_data = torch.tensor(tmp_y, \
      dtype=torch.float64).to(device)
    self.all_data = torch.tensor(all_xy, \
      dtype=torch.float64).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    densities = self.x_data[idx,:]  # or just [idx]
    coef = self.y_data[idx,:] 
    return (densities, coef)       # tuple of two matrices 

# 3. Class to Convert Data to Torch Tensors
class Data(Dataset):
  def __init__(self, X, y):
    self.X = torch.from_numpy(X.astype(np.float64))
    self.y = torch.from_numpy(y.astype(np.float64))
    self.len = self.X.shape[0]

    
    self.x_data = torch.tensor(self.X, \
      dtype=torch.float64).to(device)
    self.y_data = torch.tensor(self.y, \
      dtype=torch.float64).to(device)

      
  def __getitem__(self, index):
    return self.X[index], self.y[index]
  
  def __len__(self): 
    return self.len
  



# DATA PREPARATION OF ONE DATASET USING FITTED SCALERS FROM ANOTHER DATASET (avoid data leakage)-----------------
def normalize_dataset(train_data, test_data, val_data, scalers, skewed_features):
  # 1. Perform copies to avoid modifying the original data
  train_data_transformed = (train_data).clone() 
  test_data_transformed  = (test_data).clone()
  val_data_transformed   = (val_data).clone()
  
  # 2. Apply the log transform to the skewed features
  if len(skewed_features) > 0:    
    train_data_transformed[:, skewed_features] = torch.log1p(train_data_transformed[:, skewed_features])
    test_data_transformed[:, skewed_features] = torch.log1p(test_data_transformed[:, skewed_features])
    val_data_transformed[:, skewed_features] = torch.log1p(val_data_transformed[:, skewed_features])

  # 3. Apply the min max scaler
  train_data_scaled = torch.cat([torch.tensor(scaler.transform(train_data_transformed.cpu().numpy()[:, i:i+1])) for i, scaler in enumerate(scalers)], dim=1)
  test_data_scaled = torch.cat([torch.tensor(scaler.transform(test_data_transformed.cpu().numpy()[:, i:i+1])) for i, scaler in enumerate(scalers)], dim=1)
  val_data_scaled = torch.cat([torch.tensor(scaler.transform(val_data_transformed.cpu().numpy()[:, i:i+1])) for i, scaler in enumerate(scalers)], dim=1)

  return np.array(train_data_scaled), np.array(test_data_scaled), np.array(val_data_scaled)

# train-test-split methods
def setup_dataset_with_preproprocessing_info(X_data, y_data, preprocessing_info):

  # 1. Train-test-validation split
  train_size = int(preprocessing_info.fraction_train * len(X_data)) 
  val_size = int(preprocessing_info.fraction_val * len(X_data)) 
  test_size = len(X_data) - train_size - val_size 
  X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=test_size, random_state=4) 
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=4) 
  
  # 2. extract information about the preprocessing 
  scalers_output      = preprocessing_info.scalers_output # fitted min max scalers 
  scalers_input       = preprocessing_info.scalers_input  # fitted min max scalers 
  skewed_features_out = preprocessing_info.skewed_features_out
  skewed_features_in  = preprocessing_info.skewed_features_in
  
  # 3. apply normalization 
  y_train_norm, y_test_norm, y_val_norm = normalize_dataset(torch.tensor(y_train), torch.tensor(y_test), torch.tensor(y_val), scalers_output, skewed_features_out)
  X_train_norm, X_test_norm, X_val_norm = normalize_dataset(torch.tensor(X_train), torch.tensor(X_test), torch.tensor(X_val), scalers_input, skewed_features_in)

  # 4. Convert Data to Torch Tensors
  train_data = Data(X_train_norm, y_train_norm)
  test_data  = Data(X_test_norm, y_test_norm)
  val_data   = Data(X_val_norm, y_val_norm)

  return train_data, test_data, val_data