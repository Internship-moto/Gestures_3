#Import main libraries
import pandas as pd
import numpy as np

# graphic libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys

# Import custom functions
sys.path.insert(1, '../')

from utils.functions import config_reader#, get_nogo, postporocessing_nogo

# import constants from the config
config = config_reader('../config/data_config.json') 




def get_signals_plot(
    X_train:np.array, y_train:np.array, GLOVE_CH=config.GLOVE_CH, title:str=None, plot_counter:int=None):
    """Displays free movements plot (done with no protocol)

    Args:
        X_train (np.array): Train data
        y_train (np.array): targets. Default: 6 dependent variables
        GLOVE_CH (list) - target names
        title (str): figure title
        plot_counter(int) - figure number
    """    

    GLOVE_CH = GLOVE_CH[:-1]  # Limit sensors number to 5
    dist = - np.arange(len(GLOVE_CH)) * 200 # display distanced labels

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].plot(X_train)
    ax[0].set_title('OMG')
    
    ax[1].plot(y_train + dist)  
    ax[1].yaxis.set_ticks(dist, labels=GLOVE_CH)
    ax[1].set_title('Glove')
    
    
    if title is not None:
        fig.suptitle(title, y=0.01, fontsize=14)
   
    
    if plot_counter is not None:
        plt.savefig(f'../figures/fig_{plot_counter}.png')
    
    plt.tight_layout()
    #plt.show();

  
    

def get_signals_comparison_plot( 
    y_test:np.array, y_pred:np.array,
    y_train:np.array=None, y_pred_train:np.array=None, 
    GLOVE_CH=config.GLOVE_CH,
    plot_counter:int=None): #, only_test:bool=True
    """Displays test and predicted data on the same plot

    Args:
        y_train (np.array): targets (dependent variables) of the train sample
        y_test  (np.array): targets (dependent variables) of the test sample
        y_pred  (np.array): predicted values for the test sample
        GLOVE_CH (list) - target names
        plot_counter (int) - figure number.                              
        
    """    
    fig, ax = plt.subplots(1, 1, figsize=(10, 4)) # plt.sca(axes)

    GLOVE_CH = GLOVE_CH[:-1]
    y_test = y_test  # multiplication of test values
    
    # Слагаемые для разделения показаний датчиков
    yticks = np.arange(len(GLOVE_CH)) * 200
    lines, labels = [], []
    indexes = np.arange(y_test.shape[0])
    
    # Display only test data
    if y_train is None:

        p = plt.plot(indexes, np.subtract(y_test, yticks), c='C0')
        lines += [p[0]]
        #labels += ['y_true']

        p = plt.plot(indexes, np.subtract(y_pred, yticks), c='C1', linestyle='-')
        lines += [p[0]]
        #labels += ['y_pred']
        #plt.axvline(y_train.shape[0] , color='k')
        
        labels = ['y_true', 'y_pred']
        major_ticks = np.linspace(0, round(y_test.shape[0], -2), (np.round(y_test.shape[0], -3)/1000).astype(np.int32)+1)
        ax.set_xticks(major_ticks)
        ax.xaxis.grid(linestyle='--')
        plt.yticks(-yticks, GLOVE_CH)
        plt.legend(lines, labels)
        plt.suptitle(f'Gestures')
        plt.tight_layout();
     
    # Display train and test data   
    else:
        #print('only_test must be True')
        
        p = plt.plot(np.arange(y_train.shape[0]), np.subtract(y_train, yticks), c='C0')
        plt.plot(y_train.shape[0] + np.arange(y_test.shape[0]), np.subtract(y_test, yticks), c='C0')
        lines += [p[0]]
        labels += ['y_true']

        p = plt.plot(np.arange(y_train.shape[0]), np.subtract(y_train, yticks), c='C1', linestyle='-')
        plt.plot(y_train.shape[0] + np.arange(y_test.shape[0]), np.subtract(y_pred, yticks), c='C1', linestyle='-')
        lines += [p[0]]
        labels += ['y_pred']
        plt.axvline(y_train.shape[0], color='k') # displays the boundary between trains and tests values

        plt.yticks(-yticks, GLOVE_CH)
        plt.legend(lines, labels)
        plt.suptitle(f'Gestures')
        plt.tight_layout()
        
    if plot_counter is not None:
        plt.suptitle(f"Fig.{plot_counter} - Истинные координаты и предсказания", y=0.05, fontsize=14)
        plt.savefig(config.PATH_FIGURES + f'fig_{plot_counter}.png')
    else: 
        plot_counter = 1
        plt.suptitle(f"Fig.{plot_counter} - Истинные координаты и предсказания", y=0.05, fontsize=14)
        
        
def plot_history(history, model_name:str=None, plot_counter:int=None):
    """Функция визуализации процесса обучения модели.
    Аргументы:
    history (keras.callbacks.History) - история обучения модели,
    model_name (str) - figure title. Use: model.name
    plot_counter (int) - порядковый номер рисунка.      
    """
    mse_metric = history.history['mse'] 
    mse_val = history.history['val_mse']  # на валидационной выборке 
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(mse_metric))

    # визуализация систем координат
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(11, 4))

    ax[0].plot(epochs, loss, 'b', label='Training loss')
    ax[0].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[0].set_xlabel('Epoch', size=11)
    ax[0].set_ylabel('Loss', size=11)
    ax[0].set_title('Training and validation loss')
    ax[0].legend()

    ax[1].plot(epochs, mse_metric, 'b', label='Training MSE')
    ax[1].plot(epochs, mse_val, 'r', label='Validation MSE')
    ax[1].set_xlabel('Epoch', size=11)
    ax[1].set_ylabel('MSE value', size=11)
    ax[1].set_title(f"Training and validation MSE")
    ax[1].legend()

    if plot_counter is not None:
        plt.suptitle(f"Fig.{plot_counter} - {model_name} learning", y=0.05, fontsize=14)
        plt.savefig(config.PATH_FIGURES + f'fig_{plot_counter}.png')
    
    else: 
        plot_counter = 1
        plt.suptitle(f"Fig.{plot_counter} - {model_name} learning", y=-0.1, fontsize=14)  
    plt.tight_layout();

    # fig.show(); #- не вызывать для корретного логгирования
    
