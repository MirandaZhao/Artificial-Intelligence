# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:17:40 2019

@author: miran
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# load dataset
dataframe = pd.read_csv('winequality-red.csv', delimiter = ",", header=None)
start_time = datetime.datetime.now()
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
dataframe = dataframe.replace(np.nan,0)
dataset = dataframe.values
data=pd.DataFrame(dataset)
#print(dataset)
#print(dataset.shape)
#print(data.head(10))


# split into input (X) and output (Y) variables
X = dataset[:,0:11]                          
Y = dataset[:,11]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)
y_binary = to_categorical(y_int)

#Normalization input data
scaler = StandardScaler()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""dataset fields 
Input variables (based on physicochemical tests): 
1 - fixed acidity 
2 - volatile acidity 
3 - citric acid 
4 - residual sugar 
5 - chlorides 
6 - free sulfur dioxide 
7 - total sulfur dioxide 
8 - density 
9 - PH 
10 - sulphates 
11 - alcohol 
Output variable (based on sensory data): 
12 - quality (score between 0 and 10)
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# define base model
# create model
model = Sequential()
model.add(Dense(11, input_dim=11, activation='relu'))
model.add(Dense(8, activation='softmax'))                                                 #Can I change neurons or just because neurons means dimension so it si fixed?          
model.add(Dense(1))                                                      
# Compile model
model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# fix random seed for reproducibility 
seed = 8
np.random.seed(seed)

estimator = model.fit(X_train, Y_train, epochs=5, verbose=1)                                     #The meaning of adding batch size here?
                                                                                    #How to define epochs and batch size?  If unspecified, batch will default to 32.


                                                                                       
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#plot metrics
pyplot.plot(estimator.history['mean_squared_error'])
pyplot.plot(estimator.history['mean_absolute_error'])                               #xy express?
pyplot.show()

#Time Required
stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

#Print out MSE
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(X_test)










