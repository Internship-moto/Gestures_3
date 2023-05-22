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


def get_all_sensors_plot(id:int, X_train:pd.DataFrame, plot_counter:int=None):
    """
    Plots sensor signals by arguments:
     
    id(int) - test number;
    X_train(pd.dataFrame) - training sample;
    plot_counter(int) - figure number.
    """
    
    fig = px.line(data_frame=X_train[id].T)
    
    fig.update_layout( 
        xaxis_title_text = 'Periods', 
        yaxis_title_text = 'Sensor signals', # yaxis_range = [0, 3000],
        legend_title_text= 'Sensor <br> index',
        width=600, height=400,  margin=dict(l=20, r=20, t=20, b=100),
    )

    fig.show()

    # # сохраним результат в папке figures. Если такой папки нет, то создадим её
    # if not os.path.exists("figures"):
    #     os.mkdir("figures")

    if plot_counter is not None:
        fig.write_image(f'../figures/fig_{plot_counter}.png', engine="kaleido")
    else: 
        plot_counter = 1
        
    fig.update_layout(title=dict(text=f'Рис. {plot_counter} - Sensor signals <br> for the test ' + str(id), x=.5, y=0.08, xanchor='center'))

def get_signals_plot(X_train:np.array, y_train:np.array, GLOVE_CH=config.GLOVE_CH, title:str=None):
    """Displays free movements plot (done with no protocol)

    Args:
        X_train (np.array): Train data
        y_train (np.array): targets. Default: 6 dependent variables
        GLOVE_CH (list) - target names
        title (str): chart title
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
        fig.suptitle(title, fontsize=14)
        
    fig.tight_layout()



def get_nogo_plot(y_train:np.array, limits:tuple, GLOVE_CH:list=config.GLOVE_CH):
    """Displays nogo status chart

    Args:
        y_train (np.array): targets. Default: 5 dependent variables
        low_lim (int): low limit
        high_lim (int): high limit
        GLOVE_CH (list, optional): Target names list. Defaults to config.GLOVE_CH.
    """    
    GLOVE_CH = GLOVE_CH[:-1]
    low_lim, high_lim = limits[0], limits[1]
    dist = -np.arange(5)*200

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6)) 
    ax[0].plot(np.arange(low_lim, high_lim), y_train[low_lim:high_lim] + dist)  
    ax[0].yaxis.set_ticks(dist, GLOVE_CH) 
    ax[0].set_title('Glove')

    ax[1].plot(np.arange(low_lim, high_lim), get_nogo(y_train)[low_lim:high_lim])
    ax[1].set_title('Nogo')
    ax[1].yaxis.set_ticks(np.linspace(0,1,2))

    ax[2].plot(np.arange(low_lim, high_lim), postporocessing_nogo(get_nogo(y_train))[low_lim:high_lim])
    ax[2].set_title('Nogo postprocessed')
    ax[2].yaxis.set_ticks(np.linspace(0,1,2));

   
    

def get_signals_comparison_plot(y_train:np.array, y_test:np.array, y_pred:np.array, GLOVE_CH=config.GLOVE_CH, only_test:bool=True):
    """Displays test and predicted data on the same plot

    Args:
        y_train (np.array): targets (dependent variables) of the train sample
        y_test  (np.array): targets (dependent variables) of the test sample
        y_pred  (np.array): predicted values for the test sample
        GLOVE_CH (list) - target names
        only_test (int, optional): a lable for displaying the given dataset [0,1]. 
                                Defaults to None and displays train and test data, 1 - display only test data
        
    """    
    fig, axes = plt.subplots(1, 1, figsize=(10, 4)) # plt.sca(axes)

    GLOVE_CH = GLOVE_CH[:-1]
    
    # Слагаемые для разделения показаний датчиков
    yticks = np.arange(len(GLOVE_CH)) * 200
    lines, labels = [], []
    indexes = np.arange(y_test.shape[0])
    
    # Display only test data
    if only_test is True:

        p = plt.plot(y_train.shape[0] + np.arange(y_test.shape[0]), np.subtract(y_test, yticks), c='C0')
        lines += [p[0]]
        labels += ['y_true']

        #p = plt.plot(y_train.index, y_train.values + yticks, c='C1', linestyle='-')
        p = plt.plot(y_train.shape[0] + np.arange(y_test.shape[0]), np.subtract(y_pred, yticks), c='C1', linestyle='-')
        lines += [p[0]]
        labels += ['y_pred']
        plt.axvline(y_train.shape[0] , color='k')

        plt.yticks(-yticks, GLOVE_CH)
        plt.legend(lines, labels)
        plt.suptitle(f'Gestures')
        plt.tight_layout()
     
    # Display train and test data   
    else:
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
        
        
def plot_history(history, title:str=None, plot_counter:int=None):
    """Функция визуализации процесса обучения модели.
    Аргументы:
    history (keras.callbacks.History) - история обучения модели,
    title (str) - figure title. Use: model.name
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
        plt.suptitle(f"Fig.{plot_counter} - Model learning",  fontsize=14) # y=-0.1,
        plt.write_image(f'../figures/fig_{plot_counter}.png', engine="kaleido") #savefig(...)
    
    else: 
        plot_counter = 1
        plt.suptitle(f"Fig.{plot_counter} - {title} learning", y=1, fontsize=14) # y=-0.1,

 
    # fig.show(); #- не вызывать для корретного логгирования