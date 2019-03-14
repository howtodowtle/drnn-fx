
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

import matplotlib.pyplot as plt
from IPython.display import Image
from keras.utils import plot_model, to_categorical

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, GRU, Dropout
from keras.layers import CuDNNLSTM, CuDNNGRU
from keras.callbacks import EarlyStopping, ModelCheckpoint


# ### Evaluation
# #### Economic evaluation

# In[13]:


def trading_strategy(y_true_returns, y_pred, midpoint=0.5, threshold=0):
    """
    Calculates cumulative absolute profits (i.e. p.a. profits for 250 days 
    of trading) from a simple trading strategy of going long when predicted 
    returns are on or above a midpoint + threshold (Default: 0.5 + 0) and 
    short when below midpoint - threshold.    
    threshold = 0 (Default) means trading every prediction.
    """
    
    returns = []
    
    for i in range(len(y_pred)):
        
        # if model predicts positive return,  go long
        if y_pred[i] >= midpoint + threshold:
            returns.append(y_true_returns[i])
            
        # else, go short
        elif y_pred[i] < midpoint - threshold:
            returns.append(0 - y_true_returns[i])
    
    profits = listproduct([(1 + r) for r in returns]) - 1
    stdev = np.std(returns)
    sharpe_ratio = np.mean(returns) / stdev
    
    return profits, stdev, sharpe_ratio


# #### Evaluation metrics

# In[14]:


def evaluation(model, time_series, x_true, y_true):
    """
    Evaluates a model's predictions.
    """
    # get true returns (i.e. not the binary variable y_true) 
    # from time_series (before scaling)
    y_trade_returns = time_series[-250:].values
    
    # predict y_trade
    y_pred = model.predict(x_true, verbose=0)[:, 0]
    
    # profits: true returns, predicted probabilities
    profits, stdev, sharpe_ratio = trading_strategy(y_trade_returns, y_pred)
    
    # log loss and accuracy: x_trade sequences, true binary labels
    log_loss, accuracy = model.evaluate(x_true, y_true, verbose=0)
    
    # area under ROC curve: true binary labels, predicted probabilities
    roc_auc = roc_auc_score(y_true, y_pred)
    
    return log_loss, accuracy, roc_auc, profits, stdev, sharpe_ratio


# ### Helper functions
# #### Listproduct

# In[15]:


def listproduct(lst):
    """
    Takes a list argument and returns the 
    product of all its elements.
    """
    product = 1
    for number in lst:
        product *= number
    return product


# #### Multi-index dataframe for results

# In[16]:


def create_results_df(study_periods, metrics, models=['FNN', 'SRNN', 'LSTM', 'GRU']):
    """
    Returns a multi-index pd.DataFrame filled with '_' in each cell.
    Columns: evaluation metrics
    Row levels: models (level 0), study periods (level 1)
    """
    
    # multi-index
    idx = pd.MultiIndex.from_product([models,  # for each model type
                                     list(range(1, study_periods + 1))],  # one row per study period
                                     names=['Model', 'Study period'])

    # empty results dataframe 
    return pd.DataFrame('-', idx, metrics)


# #### Dataframe aggregation by model

# In[17]:


def aggregate_results_by_model(granular_df, models=['FNN', 'SRNN', 'LSTM', 'GRU']):
    """
    Aggregates the values in a granular dataframe 
    (data for every study period) by model.
    """
    # aggregated per model results
    aggregated_df = pd.DataFrame(columns=metrics)

    for nn in models:
        aggregated_df.loc[nn] = granular_df.loc[nn].mean(axis=0)

    return aggregated_df


info = '\nThis module contains the following methods: \
        \n- trading_strategy(y_true_returns, y_pred, midpoint=0.5, threshold=0) \
              \n    - listproduct(lst) \
        \n- evaluation(model, time_series, x_true, y_true)\
        \n- create_results_df(study_periods, metrics, models=[\'FNN\', \'SRNN\', \'LSTM\', \'GRU\']) \
        \n- aggregate_results_by_model(granular_df, models=[\'FNN\', \'SRNN\', \'LSTM\', \'GRU\'])'


print(info)