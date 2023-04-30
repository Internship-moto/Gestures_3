# Импортируем библиотеки
import re
import os
import sys
import glob
import json
from dotmap import DotMap


# библиотека взаимодействия с интерпретатором
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def config_reader(path_to_json_conf: str) -> dict:
    """Функция загрузки параметров конфигурации в память.

    Args:
    ------------
    path_to_json_conf (_str_): путь к файлу конфигурации

    Returns:
    ------------
    config (dict): словарь с параметрами конфигурации
    """    
    with open(path_to_json_conf, 'r') as config_file:
        config_dict = json.load(config_file)
    
    config = DotMap(config_dict)
    
    return config


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

