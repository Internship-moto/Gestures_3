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
    y_test (np.array) - target values from the test sample 
    y_pred (np.array) - predictions for the test sample
    y_train (np.array) - target values from the train sample 
    y_pred_train (np.array) - predictions for the train sample
    GLOVE_CH (list) - target names
    train_test (bool) - var for displaying metrics for train and test samples. By default displays only for the test sample
    ------
    """
    GLOVE_CH = GLOVE_CH[:-1]  # Limit sensors number to 5
    metrics_test = np.array([mean_squared_error(y_test[:,col], y_pred[:,col]) for col in range(y_test.shape[1])]) # for Numpy
    
    if y_train is not None and y_pred_train is not None:
        metrics_train = np.array([mean_squared_error(y_train[:,col], y_pred_train[:,col]) for col in range(y_train.shape[1])])
        display(pd.DataFrame({'Train':metrics_train, 'Test': metrics_test}, index=GLOVE_CH))
        
    else:
        display(pd.DataFrame({'Test':metrics_test}, index=GLOVE_CH))
        
    
def get_nogo(df:np.array)->np.array:
    """Gets 'nogo' condition from all sensors

    Args:
        df (np.array): _description_

    Returns:
        np.array: _description_
    """   
    # Get diff for all sensors . Shape = (28975, 5) 
    nogo = np.diff(df, axis=0) 

    # Get changes vector. Shape = (28975, 1)
    Nogo = np.sum(np.abs(nogo), axis=1)#, keepdims=True
    
    return np.where(Nogo>1,1,0)

def postporocessing_nogo(arr:np.array)->np.array:
    """Delete sponatious single peaks from the given array
    Args:
        arr (np.array): data array
    Returns:
        np.array: _description_
    """
    # initial value
    i = 0

    for i in range(len(arr)):
        
        step = arr[i:i+4]
        try:
            # case [0,0,0,0] or [1,1,1,1]. Смещаемся на 1 элемент вправо
            if step[3]==step[2]: 
                i+=1 
                #step = arr[i:i+4]
                
            # case [0,0,0,1]. Смещаемся на 2 элемента вправо
            else:
                i+=2
                step = arr[i:i+4]
                
                #case [1,0,0,0]
                if step[1]==step[2]==step[3]:
                    step[0]==step[1]
                    i+=1
                
                #case [1,0,1,0] and [1,0,1,1]
                else:  
                    step[1]=step[0]
                    i+=1
        except:
            break
    
    return arr
            

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

