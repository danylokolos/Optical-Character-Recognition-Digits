# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:22:56 2021

@author: Danylo
"""

#%% Read in Data
import numpy as np
import pandas as pd
import pickle
import os
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

images_path = 'C:/Users/Danylo/Documents/Education-2021-SAIT-MachineLearning/Assignments/Assignment05-OpticalCharacterRecognitionDigits/InputData/train-images-idx3-ubyte/train-images.idx3-ubyte'
labels_path = 'C:/Users/Danylo/Documents/Education-2021-SAIT-MachineLearning/Assignments/Assignment05-OpticalCharacterRecognitionDigits/InputData/train-labels-idx1-ubyte/train-labels.idx1-ubyte'

X, y = loadlocal_mnist(
    images_path=images_path, 
    labels_path=labels_path)
    
#%% Data Wrangling
#none

#%% Preprocessing
#normalize data
X = X/255

# reshape
X_3d = X.reshape(60000,28,28,1)
#X = X_3d

#%% Split data
from sklearn.model_selection import train_test_split

random_state = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

"""
# define cnn model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Dense(10, activation='softmax'))
# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

model.summary()
"""


































#%% Machine Learning Model
import tensorflow as tf
from tensorflow.keras import models,layers
import time 
start = time.time()


#hidden1_num_nodes = [64,128,256,512,1024]
hidden1_num_nodes = [512]

#hidden1_activation = ['relu','sigmoid','softmax']
hidden1_activation = ['relu']

#hidden1_dropout = np.arange(0.1,1.0,0.1)
hidden1_dropout = np.arange(0.4)


ii = 1
i_total = len(hidden1_num_nodes)*len(hidden1_activation)*len(hidden1_dropout)
testing_result = pd.DataFrame()
for i_node in hidden1_num_nodes:
    for i_act in hidden1_activation:
        for i_drop in hidden1_dropout:
            print('Running Model:', ii, 'of', i_total)
            print('Current Running Time:', time.time()-start)
            ii = ii+1
            
            model = models.Sequential()
            # input layer
            model.add(layers.Dense(784, activation='relu'))
            #model.add(layers.Conv2D(32,(3,3), activation='relu',input_shape=(28,28,1)))
            # Hidden
            model.add(layers.Dense(i_node, activation=i_act))
            #model.add(layers.Dropout(i_drop))
            # Output layer
            model.add(layers.Dense(10, activation='softmax'))
            model.compile(optimizer='adam',loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=5)
            
            model.summary()
            
            model_quality = model.evaluate(X_test, y_test)
            testing_result = testing_result.append([[i_node, i_act, model_quality[0], model_quality[1],i_drop]],ignore_index=True)
                
            # save results in case of crash
            testing_result.to_csv('testing_results2.csv')

            # save model
            model_filename = "model_CNN_{}_{}_{}".format(i_node,i_act,i_drop)
            os.mkdir(model_filename)
            model.save(model_filename, '/', model_filename)

# rename columns and save
testing_result = testing_result.rename(columns={0:'hidden1_num_nodes', 1:'hidden1_activation', 2:'loss value', 3:'metrics=accuracy', 4:'Dropout'})
testing_result.to_csv('testing_results2.csv')

