# Hand prosthesis automation


## Content

* [Summary](README.md#Summary)
* [Introduction](README.md#Introduction)
* [Data and methods](README.md#Data-and-methods)  
* [Results](README.md#Results)     
* [Проверка универсальности алгоритма](README.md#Проверка-универсальности-алгоритма)
* [Заключение](README.md#Заключение)                    
* [Структура проекта](README.md#Структура-проекта)        


## Summary

It was created a custom algorthm to manage a bionic hand prosthesis using muscle signals form optomyographic sensors. The mean execution time of the program is $7\pm1$ msec that is about five time less than an aveage human is needed to react. The core part of the algorithm is an LSTM neural network which satisfactory adopts to various hand gestures and initial position of sensors.

<p align="center"> <img src = "figures/prosthesis.gif"  alt="drawing" style="width:500px;"> </p>

## Introduction

A traumatic loss or damage of a person's hand significantly limits his work ability in modern society. Even daily routine task like brushing teeth or grabbing a door handle can be slower. Although passive or body powered prostheses may add some freedom, their grippers lack universiality, and their construction restricts fine movements. Furtheromore, actuation systems of these types of prostheses may cause discomfort. As a result, passive and semi-active devices still limit the work capacity of an impaired person. By contrast, bionic prostheses, which allow independent moves of its fingers, are more compact, but could be considerably heavier because of accumulators. Nevertheless, their microcontroller-based actuating systems allow operators to make wider range of movements that assures better user experience. 

Since each traumtic incident is unique, the variation of hand disabilities is relatively broad. In this study, we consider the case of traumatic palm loss without severe affection to the forearm's muscules actuating fingers. This biophysical condition fits best with the rehabilitation with a non-invasive bionic hand prosthesis. 

From the engineering point of view, a bionic prosthesis can be represented as a combination of three main systems: mechanical, electrical, and operational. The last system is in our particular interst because the prosthesis must imitate a natural biophysical function, and that is a difficult task of following reasons. On first, transmission of neuro signals through human nervs goes almost without delays. Any external system must be fast enough to assure positive user experience from its usage. On second, hand manipulation ususally involves different muscles, and each of them may change its state while performing a fine movement. Thus, the system must input a complex combination of neuro signals and output a coordinate of finger or fingers. On third, the displacement of muscules and their size of is unique for any forearm. This means prosthesis' sensors must percept neuro sygnals from certain areas of forearm rather than points. Taken into account these circumstances, a AI-based algorithm is the best solution for such a complex task.

Thus, the project goal was to develop an autoamtion algorithm for a bionic hand prosthesis that predicts finger coordinates in the range from 0 to 100 based on signals from optomyographic sensors. The operating time of 1 cycle of the algorithm should be less than 33 ms. In addition, it is necessary to check the algorithm's adaptivity to various montages of the prosthesis.



## Data and methods

As input data, we used signals from forearm muscles that were aquired by a following standardized protocol: an experimenter (having two healthy hands) had been performed a repeating sequence of gestures. The movements set consists of the most frequently encountered finger gestures in everyday life: grip, open palm, “pistol”, bend of the thumb, etc. Muscle tension was recorded by 50 optomyographic (OMG) sensors with the frequency of 30.3 Hz. The total recorded time was slightly less than 30 minutes. Every 10 minutes the experimenter was changing the orientation of the inner palm side: 0 degrees (parallel to the ground), 90 degrees, and -90 degrees. The finger coordinates, their acceleration and the fingers' angles were measured by an additional sensors, accelerometers and hyroscopes respectively. All these detectors were attached to the special glove, which analog is siaplayed on the Fig. 1.

<p align="center"> <img src = "figures/fig_1_Glove_en.png"  alt="drawing" style="width:425px;"> </p>

The experimental set of discrete signals was originally recorded in the *.palm* format and converted into the *.csv* format using the *Pandas* library and the *MNE* open library. A sample of muscle signal and fingrer coordinates is shown on the *Fig.2*.
 
<center> <img src = "figures/fig_2_en.png"  alt="drawing" style="width:800px;"> </center>

During the signals analysis we faced with multiple challenges:
* the signals amplitude of the same gestures changes over time. This might be happened due to fatigue of the experimenter.
* fingers move together while performing most of gestures. This dynamics was caused by natural structure of human hands and inherited characteristics of ligaments.
* tension and relaxation of the forearm muscles during the gesture may occur unevenly in time. This may also be explained by natural structure of human hand.
* it is difficult to repeat a sensor installation when OMG sensors are placed on the same area of the forearm. The detection area is relatively small whereas the muscle surface is uneven. As a result, it is hard to find the same muscle area.

These circumstances limit consecutive combination of raw signals from different montages. Thus, we decided to modulate the frequency of the second montage before its join to the first montage (see Fig. 3).  

<p align="center"> <img src = "figures/fig_3_montage_en.png"  alt="drawing" style="width:300px;"> </p>

## Results

To predict finger coordinates we designed an algorithm that consist of four logical blocks:
* Signal preprocessing
* Creation of the ML model
* Postprocessing of predic
* Command forming

Algorithm's scheme is shown on the fig.4.

<p align="center"> <img src = "figures/fig_4_shceme_en.png"  alt="drawing" style="width:500px;"> </p>

The data preprocessing includes several consequtive steps. On first, the raw signals need scaling to the range $r \in [0,1]$. The outliers falling outside the ranage of $\pm 1.5 \cdot IQR$ are  replaced with the boundary values. After this, the data array is combined with an array of descending differences between the current signal value and the value for the 5th past period.

The combined array of preprocessed data is first divided into segments of 40 periods to exclude training of the model for the protocol, which is confirmed by the training schedule (see Fig. 5). Then training and test samples are selected from the array. To predict coordinates, a neural network model with an LSTM layer is used, taking into account the nonlinear nature of OMG signals. The coefficients of the trained model are saved.

<p align="center"> <img src = "figures/fig_5_en.png"  alt="drawing" style="width:800px;"> </p>

To speed up the prediction of our model after deployment to the prosthesis, we used two important method. On firs, we selected a statefull model architecture, which buffers tensors. In addition, we transfered it the weights, which we obtained after model training.
As a result, the total responce time of our algorithm on the test data lies within the range $7 \pm 1 \text { milli second}$ that is almost four times fewer than our limit of 33 milli second. 

<p align="center"> <img src = "figures/fig_6.png"  alt="drawing" style="width:500px;"> </p>

The lack of experimental data and the requirement to use algorithms with simple architecture caused that out model appeared quite sensible, which is affects the accumualtor charge of the prosthesis. In fact, its prediction was quite noisy that led to unwanted movements of artificial fingers. To elliminate these undesirable effect, we applied a Holt-Winters filter, which does exponential smoothing of predicted coordinates following the formula: 

$$s_{t}=s_{t-1} + \alpha \cdot(x_{t}-s_{t-1})$$ 

where: 
* $\alpha$ - a smoothing coefficient within the range $\alpha \in [0,1]$
* $s_{t}$ and $s_{t-1}$ - the coordinate value for the period $t$ and $t-1$ respectively.

Tests have shown, that one filter is not enough to confront with unwanted finger movements. So we added a new condition, which allows movements if the predicted coordinate exceeds previous by at least 2 units. Although the measure has successfully elliminated the wrong tremor, the system appeared to be less sensible to slow movements. 

<p align="center"> <img src = "figures/fig_78.png"  alt="drawing" style="width:1000px;"> </p>

## Проверка универсальности алгоритма

Одной из задач исследования являлось получение универсального алгоритма, предсказывающего координаты пальцев на монтаже с относительно произвольным расположением ОМГ датчиков. На данных 1 монтажа был проверен следующий метод преобразования: 
* определены номера "активных" датчиков, средние уровни сигналов которых превышают 200 единиц. Для таких датчиков вычислены средние значения. 
* получено отношение средних значений сигналов для обоих монтажей
* данные 1 монтажа домножались на кооэффициент отношения средних значений сигналов активных датчиков.

Использованный метод преобразования частично повторяет последовательность движений для указательного и среднего пальцев (ENC2 и ENC3 соответственно), однако среднеквадратическая ошибка предсказания высокая.

<p align="center"> <img src = "figures/fig_9.png"  alt="drawing" style="width:1000px;"> </p>


## Заключение
Cоздан алгоритм расчёта координат пальцев для протеза кисти на основе нейросети LSTM. Достоиснства программного решеня заключаются в малом времени вычисления, малой среднеквадратической ошиби предсказания модели. Предложенный алгоритм обладает удовлетворительной адаптивностью к произвольному монтажу датчиков.


## Структура проекта

<details>
  <summary>display project structure </summary>

```Python
├── .gitignore
├── config 
│   └── data_config.json            # configuration settings
├── data                            # raw data directory
│   ├── sprint3_pilote1_mount1_gestures.palm
│   └── sprint3_pilote1_mount2_gestures.palm
├── figures                         # figures directory
│   ├── fig_1_Glove_art.PNG
│   ├── fig_2.png
│   ├── fig_3_montage.png
│   ├── fig_4_shceme.png
│   ├── fig_5_outliers.png
│   ├── fig_6.png
│   ├── fig_7.png
│   ├── fig_8 commands.png
│   └── fig_9.png
├── models                          # models and weights
│   ├── MinMax_scaler.bin
│   └── model_LSTM.hdf5
├── notebooks                       # notebooks d 
│   └── Gestures_3.ipynb
├── README.md
└── utils                       # functions directory
    ├── config_reader.py
    ├── figures.py
    ├── functions.py
    └── __pycache__
```
</details>