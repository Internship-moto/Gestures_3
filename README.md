# Hand prosthesis automation


## Content

* [Summary](README.md#Summary)
* [Introduction](README.md#Introduction)
* [Data and methods](README.md#Data-and-methods)  
* [Results](README.md#Results)     
* [Adaplivity check](README.md#Adaplivity-check)
* [Заключение](README.md#Заключение)                    
* [Project structure](README.md#Project-structure)        


## Summary

It was created a custom algorthm to manage a bionic hand prosthesis using muscle signals form optomyographic sensors. The mean execution time of the program is $7\pm1$ msec that is about five time less than an aveage human is needed to react. The core part of the algorithm is an LSTM neural network which satisfactory adopts to various hand gestures and initial position of sensors.

<p align="center"> <img src = "figures/prosthesis.gif"  alt="drawing" style="width:500px;"> </p>

## Introduction

A traumatic loss or damage of a person's hand significantly limits his work ability in modern society. Even daily routine task like brushing teeth or grabbing a door handle can be slower. Although passive or body powered prostheses may add some freedom, their grippers lack universiality, and their construction restricts fine movements. Furtheromore, actuation systems of these types of artificial limbs may cause discomfort. As a result, passive and semi-active devices still limit the work capacity of an impaired person. By contrast, bionic prostheses, which allow independent moves of its fingers, are more compact, but could be considerably heavier because of accumulators. Nevertheless, their microcontroller-based actuating systems allow operators to make wider range of movements that assures better user experience. 

Since each traumtic incident is unique, the variation of hand disabilities is relatively broad. In this study, we consider the case of traumatic palm loss without severe affection to the forearm's muscules actuating fingers. This biophysical condition fits best with the rehabilitation with a non-invasive bionic hand prosthesis. 

From the engineering point of view, a bionic prosthesis can be represented as a combination of three main systems: mechanical, electrical, and operational. The last system is in our particular interst because the prosthesis must imitate a natural biophysical function, and that is a difficult task of following reasons. On first, transmission of neuro signals through human nervs goes almost without delays. Any external system must be fast enough to assure positive user experience from its usage. On second, hand manipulation ususally involves different muscles, and each of them may change its state while performing a fine movement. Thus, the system must input a complex combination of neuro signals and output a coordinate of finger or fingers. On third, the location of muscules is unique for any forearm. Furthermore, the prosthesis has some flexibility on the forearm, so it does not necessairy reach the same musles after reattachment. This means the prosthesis' sensors must percept neuro sygnals from certain areas of forearm rather than points. Taken into account these circumstances, the best solution for such a complex signal analysis task could be an AI-based algorithm.

Thus, the project goals can be formulated as follows. On first, we must develop an autoamtion algorithm for a bionic hand prosthesis that predicts finger coordinates by the state of forearm musles. Taken into account the muscle motility under skin, the algorithm must be agnostic to various turns of the forerm.

## Data and methods

The data aquisition procedure had certain limitations. The state of muscles can be recorded only from a person with a healthy upper limbs. A disable person with the absent hand does not fit to this process because he or she cannot confirm what movements he executed. In addition, the experiment covers a set of gestures, which human the most frequently does in everyday life such as grip, open palm, “pistol”, bend of the thumb, etc. To simplify the prediction, it was created a standardized gesture protocol: an experimenter had been performing the repeatitive sequence of gestures. Every 10 minutes he turned his palm, so its angle changed from $0$ degrees (parallel to the ground) to $90$ degrees (turned to the right), and $-90$ degrees (turned to the left). The total recording period was restricted to about 30 minutes, because the pilot loses precision in gestures execution over time.

Since the grounded true data come from two different sources such as forearm and hand, their states were recorded by two separate sensing devices working in pair. The muscle condition was estimated with a custom-made bangage with a number of *optomyographic* (OMG) sensors. These 50-channels sensors precisely measured contraction or stretching of forearm muscles during hand movement with the frequency of 30.3 Hz. Another device, which was used for monitoring of hand gestures, looked like a glove to carry sensors of different types. Five *flexion sensors* estimated the angles of every phalange. Three *accelerators* monitored the speed of fingers during gestures. *Hyroscopes* recorded the orienation of palm. An a analog of the glove is displayed on the Fig. 1.

<p align="center"> <img src = "figures/fig_1_Glove_en.png"  alt="drawing" style="width:425px;"> </p>

Following the recording, the data undergo some transformation for convenience. The raw data was originally recorded in a rare  *.palm* format. So it was converted into a *.csv* format, which is a standard for such data cnalysis tasks, using the *Pandas* and the *MNE* open libraries. In addition, discrete digital signals from hyroscopes were transformed within the range from $-100$ to $100$; other signals - from $0$ to $100$. A sample of resulting signal and fingrer coordinates are displayed on the *Fig.2*.
 
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
* Postprocessing of the prediction
* Command forming

Algorithm's scheme is shown on the fig.4.

<p align="center"> <img src = "figures/fig_4_shceme_en.png"  alt="drawing" style="width:500px;"> </p>

The data preprocessing includes several consequtive steps. On first, the raw signals need scaling to the range $r \in [0,1]$. The outliers falling outside the ranage of $\pm 1.5 \cdot IQR$ are  replaced with the boundary values. After this, the data array is combined with an array of descending differences between the current signal value and the value for the 5th past period.

The combined array of preprocessed data is first divided into segments of 40 periods to exclude training of the model for the protocol, which is confirmed by the training schedule (see Fig. 5). Then training and test samples are selected from the array. To predict coordinates, a neural network model with an LSTM layer is used, taking into account the nonlinear nature of OMG signals. The coefficients of the trained model are saved.

<p align="center"> <img src = "figures/fig_5_en.png"  alt="drawing" style="width:800px;"> </p>

To speed up the prediction of our model after deployment to the prosthesis, we used two important method. On firs, we selected a statefull model architecture, which buffers tensors. In addition, we transfered it the weights, which we obtained after model training.
As a result, the total responce time of our algorithm on the test data lies within the range $7 \pm 1 \text {ms}$ that is almost four times fewer than our limit of $33 \text { ms}$. 

<p align="center"> <img src = "figures/fig_6.png"  alt="drawing" style="width:500px;"> </p>

The lack of experimental data and the requirement to use algorithms with simple architecture caused that out model appeared quite sensible, which is affects the accumualtor charge of the prosthesis. In fact, its prediction was quite noisy that led to unwanted movements of artificial fingers. To elliminate these undesirable effect, we applied a Holt-Winters filter, which does exponential smoothing of predicted coordinates following the formula: 

$$s_{t}=s_{t-1} + \alpha \cdot(x_{t}-s_{t-1})$$ 

where: 
* $\alpha$ - a smoothing coefficient within the range $\alpha \in [0,1]$
* $s_{t}$ and $s_{t-1}$ - the coordinate value for the periods $t$ and $t-1$ respectively.

Tests have shown, that one filter is not enough to confront the oversensibility of the model. So we added a new condition, which allows movements if the predicted coordinate exceeds previous by at least 2 units. Although the measure has successfully elliminated the wrong tremor, the system appeared to be less sensible to slow movements. 

<p align="center"> <img src = "figures/fig_78.png"  alt="drawing" style="width:1000px;"> </p>




## Adaplivity check

One of the tasks of current study was to ensure the algorithm may adapt well to new motages of the prosthesis. In practice, it means the system must identify "active" sensors e.g. those percepting the current state of muscles better. To realize this idea, we proposed following approach. Within every montage, we selected the first 20 sensors, whose average signal level exceeds 200 units, an arbitrary given value. The remaining "inactive" sensors outputed either relatively weak or a constant signal. In order to match signals of different montages, it was computed the total average value across  active sensors of the first montage. So, if the mean signal does not match, it will be adjusted by the coefficient. 

Overall, the proposed method showed satisfactory result on the second motage. The algorithm fairly good predicts movements for both fingers: index and middle that are represented by ENC2 and ENC3 respectively. The forecased coordinate for other fingers is worse, and their  Mean Square Error value is higher.

<p align="center"> <img src = "figures/fig_9.png"  alt="drawing" style="width:1000px;"> </p>

## Conclusion
This study explores an ai-based algorithm to automate the hand prosthesis. The algorithm's core is a machine learning models with LSTM architecture. The system reacts relatively fast, and it is relatively adaptive to new positions that artificial limb takes on the forearm.


## Project structure

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
└── utils                           # functions directory
    ├── config_reader.py
    ├── figures.py
    ├── functions.py
    └── __pycache__
```
</details>