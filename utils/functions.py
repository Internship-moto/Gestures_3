# Импортируем библиотеки
import re
import os
import sys
import glob
import json
from dotmap import DotMap
import pandas as pd

from sklearn.metrics import mean_squared_error


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




def get_mse(y_test:pd.DataFrame, y_pred:pd.DataFrame, GLOVE_CH=config.GLOVE_CH):
    """Display MSE metrics for the test sample
    
    Arguments:
    ---------
    y_test(pd.DataFrame) - target values from the test sample 
    y_pred(pd.DataFrame) - predictions
    GLOVE_CH (list) - target names
    ------
    """
    GLOVE_CH = GLOVE_CH[:-1]  # Limit sensors number to 5
    metrics_test = pd.Series({col : mean_squared_error(y_test[col], y_pred[col]) for col in GLOVE_CH})

    print('MSE metrics for Test: \n--------')
    display(metrics_test)
    
    

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

