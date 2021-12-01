# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:56:11 2021

@author: Danylo
"""

import numpy as np

#%% Load best model from OCR-Digits.py
from tensorflow.keras import models,layers
model = tf.keras.models.load_model('model_CNN_Final_v4')

#%% Run Model on Test Dataset
_y_pred = model.predict(X_test)
y_pred = np.argmax(_y_pred,axis=1)

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('========== Accuracy Score ==========')
print(accuracy)
print('========== END ==========')


# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print('========== Confusion Matrix ==========')
print(conf_matrix)
print('========== END ==========')



#%% Plot Confusion Matrix
import matplotlib.pyplot as plt

target_possibilities = ['0','1','2','3','4','5','6','7','8','9']

plt.figure(figsize = (10, 10))
cmap = plt.cm.Blues
plt.imshow(conf_matrix,cmap=cmap)
plt.grid(False)
plt.title('Handwritten Digit Confusion Matrix', size = 24)
plt.xlabel('Predicted Value',size = 20)
plt.ylabel('True Value',size = 20)
plt.colorbar(aspect=5)
#output_labels = lenc.inverse_transform(target_possibilities)
output_labels = target_possibilities

tick_marks = np.arange(len(output_labels))
plt.xticks(tick_marks,output_labels,rotation=30,fontsize='xx-large')
plt.yticks(tick_marks,output_labels,fontsize='xx-large')
for ii in range(len(output_labels)):
    for jj in range(len(output_labels)):
        if conf_matrix[ii,jj] > np.max(conf_matrix)/2:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",color="white",fontsize='xx-large')
        else:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",fontsize='xx-large')
plt.tight_layout(pad=1)
plt.savefig('Plot_ConfusionMatrix.png')



