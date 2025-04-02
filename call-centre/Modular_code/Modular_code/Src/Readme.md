# Deep Learning Time Series

Deep Learning has become a fundamental part of the new generation of Time Series Forecasting models, obtaining excellent results
While in classical Machine Learning models - such as autoregressive models (AR) or exponential smoothing - feature engineering is performed manually and often some parameters are optimized also considering the domain knowledge, Deep Learning models learn features and dynamics only and directly from the data.
Thanks to this, they speed up the process of data preparation and are able to learn more complex data patterns in a more complete way.

## MLP Model

Multilayer Perceptrons, or MLPs for short, can be applied to time series forecasting.
Univariate time series are a dataset comprised of a single series of observations with a temporal ordering and a model is required to learn from the series of past observations to predict the next value in the sequence.
Before a univariate series can be modeled, it must be prepared.
The MLP model will learn a function that maps a sequence of past observations as input to an output observation. As such, the sequence of observations must be transformed into multiple examples from which the model can learn

## CNN Model

Deep CNNs have been quite popular in areas such as Image Processing, Computer Vision, etc. Recently, the research community has been showing a growing interest in using CNNs for time-series forecasting problems.
we will proceed to understand the primary components of a Deep CNN model and understand how these components are also used for time series forecasting. Following are the primary layers of an ordinary CNN model.
1. Convolutional Layer
2. Pooling Layer
3. Fully Connected Layer


## LSTM Model

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (e.g. images), but also entire sequences of data (such as speech or video inputs).
Long Short-Term Memory Networks (LSTM) have been developed to overcome the vanishing gradient problem in the standard RNN by improving the gradient flow within the network
LSTMs can be used to model univariate time series forecasting problems.
These are problems comprised of a single series of observations and a model is required to learn from the series of past observations to predict the next value in the sequence.
The LSTM model will learn a function that maps a sequence of past observations as input to an output observation. As such, the sequence of observations must be transformed into multiple examples from which the LSTM can learn


## CNN-LSTM Model

In Machine Learning, you can now predict values on complex data by using Neural Networks. And for the majority of them, you will send one or several inputs to be analysed. Sometimes, these values are chronological. 
For example stock prices in time, video frames, or human-size at a certain age in its life. For this kind of data, we already have some nice layers to treat data in the time range, for example, LSTM. But what if you need to adapt each input before or after this layer? This is where Time Distributed layer can give a hand.



## Correlation vs AutoCorrelation

- Correlation is a bivariate analysis that measures the strength of association between two variables and the direction of the relationship. In terms of the strength of relationship, the value of the correlation coefficient varies between +1 and -1.
- A value of ± 1 indicates a perfect degree of association between the two variables. As the correlation coefficient value goes towards 0, the relationship between the two variables will be weaker.
- Auto-correlation refers to the case when your errors are correlated with each other. In layman terms, if the current observation of your dependent variable is correlated with your past observations, you end up in the trap of auto-correlation. 

## Time Series Basics

-   Chronological Data
- Cannot be shuffled
- Each row indicate specific time record
- Train – Test split happens chronologically
- Data is analyzed univariately (for given use case)
- Nature of the data represents if it can be predicted or not

## Code Description


    File Name : Engine.py
    File Description : Main class for starting different parts and processes of the lifecycle


    File Name : CNN_LSTM.py
    File Description : Code to train and visualize the CNN LSTM model
    
    File Name : CNN_Model.py
    File Description : Code to train and visualize the CNN model
    
    
    File Name : LSTM_Model.py
    File Description : Code to train and visualize the LSTM model
    
    File Name : MLP.py
    File Description : Code to train and visualize the MLP model



## Steps to Run

There are two ways to execute the end to end flow.

- Modular Code
- IPython

### Modular code

- Create virtualenv
- Install requirements `pip install -r requirements.txt`
- Run Code `python Engine.py`
- Check output for all the visualization

### IPython Google Colab

Follow the instructions in the notebook `deeplearning.ipynb`

