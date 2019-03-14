
# coding: utf-8

# # One model, one currency pair, all study periods

# ### Imports

# In[5]:


import os
import time
import warnings
import math

import numpy as np
import pandas as pd

from keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ## Function defintions

# ### Data preprocessing
# #### Scaling

# In[7]:


def scale_data(time_series, scaler, train_len):
    """
    Scales a time series.
    Scaler options: from sklearn.preprocessing: 
    MinMaxScaler (e.g. ranges(0,1), (-1,1)), StandardScaler (with or w/o std)
    Splits time series into training and trading (=test) data. 
    Scaler is fitted only to training data, 
    transformation is applied to the whole data set.
    Returns the stitched train and trade data 
    and the scaler fitted on the train data.
    """
    
    # create train and trade set
    train = time_series[:train_len]
    trade = time_series[train_len:]
    
    # scale from train set only
    # reshape to be usable with scaler
    train = train.values.reshape(train.shape[0], 1)
    trade = trade.values.reshape(trade.shape[0], 1)
    # fit scaler to training set only
    fitted_scaler = scaler.fit(train)  
    # scale both sets with the training set scaler
    train = fitted_scaler.transform(train)
    trade = fitted_scaler.transform(trade)
    # inverse transformation 
    # fitted_scaler.inverse_transform(train_z)
    # fitted_scaler.inverse_transform(trade_z)
    
    stitched = np.concatenate((train, trade), axis=0)
    
    return stitched, fitted_scaler


# #### Input sequence creation

# In[8]:


def create_input_sequences(time_series, fitted_scaler, train_len, seq_len, 
                           targets="classification_1D", pred_steps=1):
    """
    Converts a time series to a supervised problem for recurrent neural networs: 
    Creates the X's (windows of length seq_len)
    and the respective y's (the observation pred_steps steps after the windows)
    for both the train and trade set.
    targets: "regression", "classification_1D" (sparse, value in [0, 1]), 
    "classification_2D" (one-hot encoding)
    """
    
    all_windows = []  # empty time_series for training windows
    for i in range(len(time_series) - seq_len):  
        all_windows.append(time_series[i : (i + seq_len) + pred_steps])  
        # we split the windows up into (X, y) later, + pred_steps are the y's
        
    all_windows = np.array(all_windows)  # make it a numpy array
    # number of all windows = len(time_series) - seq_len
    
    split_at_row = int(train_len - seq_len)
    
    # train windows
    train_windows = all_windows[:split_at_row, :]
#     np.random.shuffle(train_windows)  
# keeps the windows intact, but shuffles their order
    
    x_train = train_windows[:, :-1]
    
    y_train = train_windows[:, -1]  # scaled returns
    if "classification" in targets:
      # one-hot encoding: col 0: returns < 0, col 1: returns >= 0
        y_train = to_categorical(fitted_scaler.inverse_transform(y_train) >= 0)  
        if targets == "classification_1D":
          # if real returns >= 0: 1, else: 0 (only take col 1)
            y_train = y_train[:, 1]  
    
    # trade windows
    trade_windows = all_windows[split_at_row:, :]
    
    x_trade = trade_windows[:, :-1]
    
    y_trade = trade_windows[:, -1]
    if "classification" in targets:
       # one-hot encoding: col 0: returns < 0, col 1: returns >= 0
        y_trade = to_categorical(fitted_scaler.inverse_transform(y_trade) >= 0) 
        if targets == "classification_1D":
          # if real returns >= 0: 1, else: 0 (only take col 1)
            y_trade = y_trade[:, 1]  

    # reshape seq.s into 3D: (samples, sequence_length/time steps, features)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 
    x_trade = np.reshape(x_trade, (x_trade.shape[0], x_trade.shape[1], 1))  

    return [x_train, y_train, x_trade, y_trade]


# #### Scaling and sequence creation

# In[9]:


def data_prep(time_series, scaler, train_len, seq_len, 
              targets="classification_1D", pred_steps=1):
    """
    Data preparation:
    Scaling, then creating input sequences for supervised learning.
    """
    # Scale time series, return scaled ts and scaler for inverse scaling
    scaled_ts, fitted_scaler = scale_data(time_series=time_series,
                                          scaler=scaler,
                                          train_len=train_len)
    
    # Create input sequences and return inputs and targets for both training and trading data
    x_train, y_train, x_trade, y_trade = create_input_sequences(time_series=scaled_ts,
                                                                fitted_scaler=fitted_scaler, 
                                                                train_len=train_len,
                                                                seq_len=seq_len, 
                                                                targets=targets, 
                                                                pred_steps=pred_steps)
    
    return [scaled_ts, fitted_scaler, x_train, y_train, x_trade, y_trade]

info = 'This module contains the following functions: \
        \n- scale_data(time_series, scaler, train_len) \
        \n- create_input_sequences(time_series, fitted_scaler, train_len, seq_len, \
                           \n    targets="classification_1D", pred_steps=1) \
        \n- data_prep(time_series, scaler, train_len, seq_len, \
        \n     targets="classification_1D", pred_steps=1)'


print(info)