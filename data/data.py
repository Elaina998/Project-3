import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mm = MinMaxScaler()
ss = StandardScaler()


def getData():
    DATAPATH = 'dataset/processed_data_best_corr_sentiment.csv'
    # Fix the file reading based on file extension
    try:
        # First, try reading as CSV since the file extension is .csv
        df = pd.read_csv(DATAPATH, index_col=0, parse_dates=True)
    except Exception as e:
        # If that fails, try with Excel with explicit engine
        try:
            df = pd.read_excel(DATAPATH, index_col=0, parse_dates=True, engine='openpyxl')
        except Exception as e2:
            # As a last resort, try with a different Excel engine
            df = pd.read_excel(DATAPATH, index_col=0, parse_dates=True, engine='xlrd')
    
    #drop the first 7 rows and obtain only 2380 to facilitate the split rate calculation
    df = df[6:]
    print(f"dataset shape: ({df.shape[0]} * {df.shape[1]}), samples * features")
    # Print column names for debugging
    print("Available columns:", df.columns.tolist())
    return df

def prepare_datat(seq_len, args):
    # read the data from the excel file
    df = getData()
    # select only two features
    df = df[args.features]
    args.feature_no = len(df.columns)
    
    # 4- X and y - Use correct column name 'BRENT Close' (with space)
    X, y = df, df['BRENT Close'].values
    # 5- Normalize
    X_trans, y_trans = normalize__my_data(X, y)

    # 7- Build the sequence
    X_ss, y_mm = split_sequences(X_trans, y_trans, seq_len, args.pred_len)

    # Calculate sizes based on 80-10-10 split
    total_samples = len(X_ss)
    train_size = int(total_samples * 0.8)
    vald_size = int(total_samples * 0.1)
    test_size = total_samples - train_size - vald_size  # Ensure all samples are used
    
    # Print split information for debugging
    print(f"Total samples: {total_samples}")
    print(f"Train size (80%): {train_size}")
    print(f"Validation size (10%): {vald_size}")
    print(f"Test size (10%): {test_size}")
    
    data = split_train_test_pred(X_ss, y_mm, train_size, vald_size, test_size)
    return data
  
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - 1
        # check if we are beyond the dataset
        if out_end_ix > len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix-1:out_end_ix, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)

def split_train_test_pred(X_ss, y_mm, train_size, vald_size, test_size):
    # Modified to use the calculated sizes rather than fixed cutoffs
    X_train = X_ss[:train_size]
    X_valid = X_ss[train_size:train_size + vald_size]
    X_test = X_ss[train_size + vald_size:train_size + vald_size + test_size]

    y_train = y_mm[:train_size]
    y_valid = y_mm[train_size:train_size + vald_size]
    y_test = y_mm[train_size + vald_size:train_size + vald_size + test_size]

    print("Training Shape:", X_train.shape, y_train.shape)
    print("Validation Shape:", X_valid.shape, y_valid.shape)
    print("Test Shape:", X_test.shape, y_test.shape)
    
    data = {
        "X_train": X_train, 
        "y_train": y_train, 
        "X_valid": X_valid, 
        "y_valid": y_valid, 
        "X_test": X_test,
        "y_test": y_test
    }
    return data

def normalize__my_data(X, y):
  X_trans = ss.fit_transform(X)
  #X_trans = x_scaler.fit_transform(X)
  y_trans = mm.fit_transform(y.reshape(-1, 1))
  return X_trans, y_trans

def denormolize_data(trues, preds):
    return mm.inverse_transform(trues), mm.inverse_transform(preds)
  
def understand_data_values_for_split(num_of_samples, X_, y_):
    """
    Split the data into training (80%), validation (10%), and test (10%) sets
    based on the percentage of data points rather than fixed dates.
    
    Args:
        num_of_samples: Number of samples to consider
        X_: Features DataFrame with datetime index
        y_: Target DataFrame with datetime index
        
    Returns:
        Training, validation, and test sets for both X and y
    """
    # Sort the dataframe by date if not already sorted
    X_ = X_.sort_index()
    y_ = y_.sort_index()
    
    # Calculate the indices for 80% and 90% points
    total_size = len(X_)
    train_end_idx = int(total_size * 0.8)
    valid_end_idx = int(total_size * 0.9)
    
    # Get the dates at these points
    all_dates = X_.index
    train_end_date = all_dates[train_end_idx]
    valid_end_date = all_dates[valid_end_idx]
    
    # Print the cutoff dates and percentages
    print(f"Total samples: {total_size}")
    print(f"Training cutoff date (80%): {train_end_date}")
    print(f"Validation cutoff date (90%): {valid_end_date}")
    
    # Split using the calculated dates
    X_train_ = X_.loc[:train_end_date]
    y_train_ = y_.loc[:train_end_date]
    
    X_valid_ = X_.loc[train_end_date:valid_end_date]
    y_valid_ = y_.loc[train_end_date:valid_end_date]
    
    X_test_ = X_.loc[valid_end_date:]
    y_test_ = y_.loc[valid_end_date:]
    
    # Print the actual sizes to verify
    print(f"Training set: {len(X_train_)} samples ({len(X_train_)/total_size:.1%})")
    print(f"Validation set: {len(X_valid_)} samples ({len(X_valid_)/total_size:.1%})")
    print(f"Testing set: {len(X_test_)} samples ({len(X_test_)/total_size:.1%})")
    
    # Visualize the split
    darw_splitted_date(num_of_samples, X_train_, X_valid_, X_test_)
    
    return X_train_, X_valid_, X_test_, y_test_, y_train_, y_valid_, y_valid_

def darw_splitted_date(num_of_samples , X_train_ , X_valid_, X_test_):
  plt.figure(figsize=(6,3))
  figure, axes = plt.subplots()
  axes.plot(X_train_["BRENT Close"][-num_of_samples:], color="blue")
  axes.plot(
      X_valid_["BRENT Close"][-num_of_samples - 259 : ],
      color="red",
      alpha=0.5,
  )
  axes.plot(
      X_test_["BRENT Close"][-num_of_samples - 259  : ],
      color="green",
      alpha=0.5,
  )
  plt.show()

def add_BRENT_Close_change(df):
  BRENT_Close_change_column="BRENT_Close_changes"
  cols= df.columns
  if BRENT_Close_change_column in cols:
    print("already added")
  else:
    df_chng= df['BRENT Close'].pct_change()
    df['BRENT Close_changes'] = df_chng
    print(df['BRENT Close_changes'].isna().sum())
    df['BRENT Close_changes'] = df['BRENT Close_changes'].replace(np.nan, 0)
  return df


from logging import raiseExceptions
# add the time/date feature to the dataset
import datetime
def add_date_features(df , date_column):
  cols=df.columns
  if date_column not in cols:
    print("the date column is unkown")
    return df
  else:
    #df_enc['dayofmonth'] = df_enc['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    #df_enc['week'] = df_enc['date'].dt.isocalendar().week
    df['season'] = (df['date'].dt.month -1)//3
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df[0:2]
    # Perform one-hot encoding of categorical date features
    encoded_data = pd.get_dummies(df, columns=['dayofweek', 'season', 'month', 'year'])
    # Display the encoded data
    return encoded_data
def variates_selection(df, date_feature_enabled, change_BRENT_Close):
  data__=[]
  data__=df.copy()
  if(change_BRENT_Close):
    data__ = add_BRENT_Close_change(data__)

  if(date_feature_enabled):
    data__= data__.reset_index()
    data__= add_date_features(data__, 'date')
    data__=data__.set_index('date')
  else:
    data__=data__.copy()
  return data__

def prepare_latest_data(features, seq_len):
    """
    Prepare the latest data for prediction
    
    Args:
        features: List of features to include
        seq_len: Length of sequence
        
    Returns:
        Dictionary with latest data sequence
    """
    # Get raw data
    df = getData()
    
    # Select only required features
    df = df[features]
    
    # Normalize data
    X, y = df, df['BRENT Close'].values
    X_trans, y_trans = normalize__my_data(X, y)
    
    # Get the latest sequence
    latest_sequence = X_trans[-seq_len:].reshape(1, seq_len, len(features))
    
    return {
        'X_test': latest_sequence,
        'y_test': np.zeros((1, 1))  # dummy value, not used for prediction
    }