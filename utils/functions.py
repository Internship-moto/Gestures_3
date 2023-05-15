# Импортируем библиотеки
import re
import os
import sys
import glob
import json
from dotmap import DotMap
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter, filtfilt, butter
import tensorflow as tf
import numpy as np
import random

# библиотека взаимодействия с интерпретатором
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
# Библиотека вызова функций, специально разработанных для данного ноутбука
sys.path.insert(1, '../')

from utils.config_reader import config_reader
from utils.figures import get_all_sensors_plot  #get_sensor_command_plot

# import constants from the config
config = config_reader('../config/data_config.json') 

def add_diff(arr:np.array, shift_=1)-> np.array:
    """Concatenation of a array values with shifted by a given step values.

    Args:
        arr (np.array): exhisting array
        shift_ (int, optional): shift of a rolling window. Defaults to 1.

    Returns:
        (np.array): concatenated array
    """    
    diff_arr = np.vstack([np.zeros((shift_, arr.shape[1])), (arr-np.roll(arr, shift_, axis=0))[shift_:]])
    return np.hstack([arr, diff_arr])

def get_mse(y_test:np.array, 
            y_pred:np.array, 
            y_train:np.array=None, 
            y_pred_train:np.array=None, GLOVE_CH=config.GLOVE_CH, displays_train_test:int=False):
    """Display MSE metrics for the test sample
    
    Arguments:
    ---------
    y_test (pd.DataFrame) - target values from the test sample 
    y_pred (pd.DataFrame) - predictions for the test sample
    y_train (pd.DataFrame) - target values from the train sample 
    y_pred_train (pd.DataFrame) - predictions for the train sample
    GLOVE_CH (list) - target names
    train_test (bool) - var for displaying metrics for train and test samples. By default displays only for the test sample
    ------
    """
    GLOVE_CH = GLOVE_CH[:-1]  # Limit sensors number to 5
    # metrics_test = pd.Series({col : mean_squared_error(y_test[col], y_pred[col]) for col in GLOVE_CH}) # for DataFrame
    metrics_test = pd.Series({col : mean_squared_error(y_test[:,col], y_pred[:,col]) for col in range(len(GLOVE_CH))}) # for Numpy
    
    if y_train is not None and y_pred_train is not None:
        metrics_train = pd.Series({col : mean_squared_error(y_train[:,col], y_pred_train[:,col]) for col in range(len(GLOVE_CH))})
        print('MSE metrics for Train: \n--------')
        display(metrics_train)
        print('MSE metrics for Test: \n--------')
        display(metrics_test)
        
    else:
        print('MSE metrics for Test: \n--------')
        display(metrics_test)
    
 
 #-------- preprocessing-------------
 
 
def preprocessing_0(col:pd.Series, window_length:int=config.window_length, polyorder:int=config.polyorder) -> pd.Series:
    """Savitsky-golay filter https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html

    Args:
        col (pd.Series): raw signals of a sensor
        window_length (int) - window length. By default window_length=60
        polyorder (int) - extent of the polynom. By default polyorder=4
    Returns:
        col (pd.Series): filtered signals of a sensor
    """    
    return savgol_filter(col, window_length, polyorder)

def preprocessing_1(b:int, a:float, col:pd.Series) -> pd.Series: 
    """digital filter forward and backward to a signal
    from https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
    Args:
        b (int): The numerator coefficient vector of the filter.
        a (float): The denominator coefficient vector of the filter. If a[0] is not 1, then both a and b are normalized by a[0]
        col (pd.Series): raw signals of a sensor

    Returns:
        filtfilt (pd.Series): filtered signals of a sensor
    """    
    bb, aa = butter(b, a)
    return filtfilt(bb, aa, col)

def reset_random_seeds(seed_value=config.seed_value):
    """Функция задания seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def callbacks(lr):
   
    # сохранение лучшей модели
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(config.PATH_MODEL, 'best_model' +'.hdf5'), 
        monitor=config.monitor, 
        verbose=1, 
        mode=config.mode, 
        save_best_only=True
    )

    # остановка обучения при отсутствии улучшения заданной метрики
    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor=config.monitor, 
        mode=config.mode, 
        patience=config.callback_patience, 
        restore_best_weights=True
    )

    # снижение learning rate при отсутствии улучшения заданной метрики 
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=config.monitor, 
        mode=config.mode,  
        factor=0.5, 
        patience=20,  # можно 10
        verbose=1, 
        min_lr=lr/1000
    )
    
    return [checkpoint, earlystop, reduce_lr]

