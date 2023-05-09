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

from utils.functions import config_reader

# import constants from the config
config = config_reader('../config/data_config.json') 


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

def get_signals_plot(X_train:pd.DataFrame, y_train:pd.DataFrame, title:str=None):
    """Displays free movements plot (done with no protocol)

    Args:
        X_train (pd.DataFrame): Train data
        y_train (pd.DataFrame): targets (6 dependent variables)
        title (str): chart title
    """    
    GLOVE_CH1 = y_train.columns.tolist()
    #GLOVE_CH1 = GLOVE_CH1[:-1]  # Limit sensors number to 5
    
    dist = -np.arange(len(GLOVE_CH1)) * 200 # display distanced labels
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].plot(X_train) # можно заменить на gestures_train['ts'].values
    ax[0].set_title('OMG')

    ax[1].plot(y_train + dist) #gestures_train[GLOVE_CH].values
    plt.yticks(dist, GLOVE_CH1)
    ax[1].set_title('Glove')
    ax[1].set_xlabel('Timesteps')
    
    if title is not None:
        fig.suptitle(title, fontsize=14)
        
    fig.tight_layout()

    
    

def get_signals_comparison_plot(y_train:pd.DataFrame, y_test:pd.DataFrame, y_pred:pd.DataFrame, only_test:int=None):
    """Displays test and predicted data on the same plot

    Args:
        y_train (pd.DataFrame): targets (dependent variables) of the train sample
        y_test  (pd.DataFrame): targets (dependent variables) of the test sample
        y_pred  (pd.DataFrame): predicted values for the test sample
        only_test (int, optional): a lable for displaying the given dataset [0,1]. 
                                Defaults to None and displays train and test data, 1 - display only test data
    """    
    fig, axes = plt.subplots(1, 1, figsize=(10, 4)) # plt.sca(axes)

    GLOVE_CH = y_train.columns.tolist()
    
    # Слагаемые для разделения показаний датчиков
    yticks = -np.arange(len(GLOVE_CH)) * 200
    lines, labels = [], []

    # Display only test data
    if only_test==1:
        #p = plt.plot(y_train.index, y_train.values + yticks, c='C0')
        p = plt.plot(y_test.index, y_test.values + yticks, c='C0')
        lines += [p[0]]
        labels += ['y_true']

        #p = plt.plot(y_train.index, y_train.values + yticks, c='C1', linestyle='-')
        p = plt.plot(y_test.index, y_pred.values + yticks, c='C1', linestyle='-')
        lines += [p[0]]
        labels += ['y_pred']
        plt.axvline(y_train.index.values[-1], color='k')

        plt.yticks(yticks, GLOVE_CH)
        plt.legend(lines, labels)
        plt.suptitle(f'Gestures')
        plt.tight_layout()
     
    # Display only train and test data   
    else:
        p = plt.plot(y_train.index, y_train.values + yticks, c='C0')
        plt.plot(y_test.index, y_test.values + yticks, c='C0')
        lines += [p[0]]
        labels += ['y_true']

        p = plt.plot(y_train.index, y_train.values + yticks, c='C1', linestyle='-')
        plt.plot(y_test.index, y_pred.values + yticks, c='C1', linestyle='-')
        lines += [p[0]]
        labels += ['y_pred']
        plt.axvline(y_train.index.values[-1], color='k') # displays the boundary between trains and tests values

        plt.yticks(yticks, GLOVE_CH)
        plt.legend(lines, labels)
        plt.suptitle(f'Gestures')
        plt.tight_layout()
        
        
def plot_history(history, plot_counter:int=None):
    """Функция визуализации процесса обучения модели.
    Аргументы:
    history (history) - история обучения модели,
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
        fig.suptitle(f"Fig.{plot_counter} - Model learning", y=-0.1, fontsize=14)
        fig.write_image(f'../figures/fig_{plot_counter}.png', engine="kaleido") #savefig(...)
    else: 
        plot_counter = 1
        fig.suptitle(f"Fig.{plot_counter} - Model learning", y=-0.1, fontsize=14)
 
    # fig.show(); #- не вызывать для корретного логгирования