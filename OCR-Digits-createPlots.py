# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:56:11 2021

@author: Danylo
"""

#%% Load best model from OCR-Digits.py
import pickle
infile = '.pkl'
model = pickle.load(infile)


#%% Run Model on Test Dataset
y_pred = model.predict(X_test)

