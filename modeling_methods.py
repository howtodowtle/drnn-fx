
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

# ### Model building functions
# #### Feedforward network (Multi-layer perceptron)

# In[10]:


def build_fnn(input_dim, hidden_layers, neurons, dropout, loss, 
              output_activation, optimizer='adam', summary=False):
    """
    Builds a feedforward neural network model for binary classification.
    input_dim: number of observations (for comparison with RNNs: sequence length)
    """
    
    model = Sequential()
    
    # input dropout
    model.add(Dropout(dropout))
        
    # first hidden layer
    model.add(Dense(neurons, activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout))

    # hidden layers in between
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation='relu'))
        model.add(Dropout(dropout))
    
    # output layer
    model.add(Dense(1, activation=output_activation))

    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=['binary_accuracy'])
    
    # print summary of layers and parameters
    if summary:
        model.summary()

    return model


# #### Recurrent Neural Networks (Simple RNN, LSTM, GRU)

# In[11]:


def build_rnn(rnn_type, input_shape, hidden_layers, neurons, dropout, loss, 
              output_activation, optimizer='adam', summary=False):
    """
    Defines a recurrent neural network model for binary classification.
    input_shape: (sequence length, number of features)
    rnn_type: RNN, SimpleRNN, LSTM, GRU, CuDRNNLSTM, CuDRNNGRU, Bidirectional 
    (see https://keras.io/layers/recurrent/)
    """
    
    model = Sequential()
    
    # input dropout
    model.add(Dropout(dropout))
    
    if hidden_layers > 1:
        # first hidden layer
        model.add(rnn_type(neurons, 
                           input_shape=input_shape, 
                           return_sequences=True))
        model.add(Dropout(dropout))
        
        # hidden layers in between
        for _ in range(hidden_layers - 2):
            model.add(rnn_type(neurons, 
                               return_sequences=True))
            model.add(Dropout(dropout))
        
        # final hidden layer before dense layer
        model.add(rnn_type(neurons))
        model.add(Dropout(dropout))
        
    else:
        # single hidden layer
        model.add(rnn_type(neurons, 
                           input_shape=input_shape))
        model.add(Dropout(dropout))
    
    # output layer
    model.add(Dense(1, activation=output_activation))

    model.compile(loss=loss, 
                  optimizer=optimizer, 
                  metrics=['binary_accuracy'])
    
    # print summary of layers and parameters
    if summary:
        model.summary()

    return model


# ### Model Training

# In[12]:


def training_one_period(model, x_train, y_train, batch_size, max_epochs=100, 
             val_split=0.2, verbose=1, patience=10):
    """
    Takes a compiled model and trains it on training data.
    """
    start_training = time.time()

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=patience, 
                               verbose=verbose, 
                               mode='auto', 
                               restore_best_weights=True)]  #,
#                  ModelCheckpoint(monitor='val_loss', 
#                                  filepath='weights/weights_e{epoch:02d}-vl{val_loss:.4f}.hdf5',
#                                  verbose=verbose, 
#                                  save_best_only=True,
#                                  save_weights_only=True)] 

    hist = model.fit(x_train, y_train, 
                     batch_size=batch_size, 
                     epochs=max_epochs,
                     verbose=verbose,
                     validation_split=val_split, 
                     callbacks=callbacks)
    
    training_time = time.time() - start_training

    if verbose > 0:
        print(f"Time: {round(training_time/60)} minutes")
    
    return model, training_time


# In[36]:


def training_all_periods(currency, hidden_layer_type, hidden_layers, neurons, dropout):
    """
    Training function that enables to specify a model to train by the currency, the type 
    and number ofhidden layers, number of neurons per hidden layer and dropout rate.
    ATTENTION:
    This function was defined mainly for readability of the later training process
    but makes use of variables defined outside the function and is not very generic.
    Use only after defining the scaler, train_len, trade_len, etc.
    """

    # timing
    start_all = time.time()

    # isolate currency pair
    ts = returns[currency].dropna()

    # determine number of study perios
    study_periods = int((len(ts) - train_len) / trade_len)

    # loop through all study periods
    for period_no in reversed(range(study_periods)):

        #### Data Preprocessing ####

        # isolate study period
        sp_stop = len(ts) - period_no * trade_len
        sp_start = sp_stop - (train_len + trade_len)
        time_series = ts[sp_start : sp_stop]

        # data preparation: scaling and creating a supervised problem
        scaled_ts, fitted_scaler, x_train, y_train, x_trade, y_trade = data_prep(time_series=time_series,
                                                                                 scaler=scaler, 
                                                                                 train_len=train_len, 
                                                                                 seq_len=sequence_len, 
                                                                                 targets='classification_1D', 
                                                                                 pred_steps=1)

        if hidden_layer_type == Dense:

            #### FNN ####

            # model building
            model_to_train = build_fnn(input_dim=sequence_len,
                                       hidden_layers=hidden_layers, 
                                       neurons=neurons, 
                                       dropout=dropout, 
                                       loss='binary_crossentropy', 
                                       output_activation='sigmoid', 
                                       optimizer='adam', 
                                       summary=False)

            # flatten input arrays (FNNs don't use sequences as inputs)
            x_train_flat = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
            # also flat, but generic name for generic evaluation function
            x_trade = np.reshape(x_trade, (x_trade.shape[0], x_trade.shape[1]))

            # model training
            trained_model, training_time = training_one_period(model_to_train, 
                                                                 x_train_flat,  # flat!
                                                                 y_train, 
                                                                 batch_size=batch_size, 
                                                                 max_epochs=max_epochs, 
                                                                 val_split=validation_split, 
                                                                 verbose=verbose, 
                                                                 patience=patience)

        else:

            #### Recurrent ####

            # rnn model
            model_to_train = build_rnn(rnn_type=hidden_layer_type,
                                       input_shape=(sequence_len, 1),
                                       hidden_layers=hidden_layers, 
                                       neurons=neurons, 
                                       dropout=dropout, 
                                       loss='binary_crossentropy', 
                                       output_activation='sigmoid', 
                                       optimizer='adam', 
                                       summary=False)

            # model training    
            trained_model, training_time = training_one_period(model_to_train, 
                                                                 x_train, 
                                                                 y_train, 
                                                                 batch_size=batch_size, 
                                                                 max_epochs=max_epochs, 
                                                                 val_split=validation_split, 
                                                                 verbose=verbose, 
                                                                 patience=patience)

        if (period_no + 1) % 5 == 0:
            print(f'{currency}, period {period_no + 1}/{study_periods}, {round(training_time/60, 1)} min')

        # get: log_loss, accuracy, roc_auc, profits, sharpe ratio
        results_this_model = list(evaluation(model=trained_model,
                                             time_series=time_series,
                                             x_true=x_trade,  # flat!
                                             y_true=y_trade))

        # append training time
        results_this_model.append(training_time)

        # write results into dataframe
        for j in range(len(metrics)):
                results_dict[currency].loc[(model_str, period_no + 1), metrics[j]] = results_this_model[j]


    # total training time for all study periods
    print(f'Done. Time: {round((time.time() - start_all)/60)} minutes')

    return results_dict



info = '\nThis module contains the following methods: \
        \n- build_fnn(input_dim, hidden_layers, neurons, dropout, loss, \
              \n    output_activation, optimizer=\'adam\', \
                \n    summary=False)dataset_raw (pd.DataFrame) \
        \n- build_rnn(rnn_type, input_shape, hidden_layers, neurons, dropout, loss, \
              \n    output_activation, optimizer=\'adam\', summary=False \
        \n- training_one_period(model, x_train, y_train, batch_size, max_epochs=100, \
             \n    val_split=0.2, verbose=1, patience=10) \
        \n- training_all_periods(currency, hidden_layer_type, hidden_layers, neurons, dropout)'


print(info)