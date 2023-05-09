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



def get_mse(y_test:pd.DataFrame, y_pred:pd.DataFrame, 
            y_train:pd.DataFrame=None, y_pred_train:pd.DataFrame=None, GLOVE_CH=config.GLOVE_CH, displays_train_test:int=False):
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
    metrics_test = pd.Series({col : mean_squared_error(y_test[col], y_pred[col]) for col in GLOVE_CH})
    
    if y_train is not None and y_pred_train is not None:
        metrics_train = pd.Series({col : mean_squared_error(y_train[col], y_pred_train[col]) for col in GLOVE_CH})
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

# def get_id_from_data():
#     """Функция загрузки номеров пилотов из данных в папке data

#     Returns:
#         id_pilot_numb_list (_int_): список с номерами пилотов
#     """    
#     id_pilot_numb_list = [] 
#     pattern = r'\d+'
#     pattern_2 = 'y_train_'

#     X_train_list = glob.glob('data/X_train_*.npy')
#     files_list = os.listdir('data')
    
#     for item in X_train_list:
#         id_pilot_num = re.search(pattern, item)[0]
#         if pattern_2 + id_pilot_num + '.npy' in files_list:
#             id_pilot_numb_list.append(int(id_pilot_num))
    
#     return id_pilot_numb_list

