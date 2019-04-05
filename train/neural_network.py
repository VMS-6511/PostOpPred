#Author: Vinith Menon Suriyakumar, Christina Yan, Mike Kennelly
#NetID: 13vms1, 14cy5, 13mwjk
#Course: CISC 452, Winter 2019

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import ast

## Both of these random seeds are set to improve the reproducibility of results
from numpy.random import seed
seed(1)

from tensorflow import set_random_seed
set_random_seed(2)

outcomes_train_df = pd.read_csv('../preprocessing/outcomes_data_train.csv')
outcomes_test_df = pd.read_csv('../preprocessing/outcomes_data_test.csv')

# Different attributes could be selected for training to determine which features provided the best classification results
X_train = outcomes_train_df[['DGN', 'PRE6',  'PRE14', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19',
              'PRE25', 'PRE30', 'PRE32', 'PRE4']].values
y_train = outcomes_train_df[['Risk1Yr']].values

X_test = outcomes_test_df[['DGN', 'PRE6',  'PRE14', 'PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19',
              'PRE25', 'PRE30', 'PRE32', 'PRE4']].values
y_test = outcomes_test_df[['Risk1Yr']].values

# This function converted the data from the CSV to a numpy array which is used for training the neural network model
def convert_to_numpy(dataframe):
    new_dataframe = []
    for row in dataframe.tolist():
        for value in range(len(row)):
            print(isinstance(row[value], str))
            if isinstance(row[value], str):
                row[value] = ast.literal_eval(row[value])
        new_row = []
        for element in row:
            if isinstance(element,list):
                for value in element:
                    new_row.append(value)
                continue
            new_row.append(element)
        new_dataframe.append(new_row)
    return np.asarray(new_dataframe)

X_train = convert_to_numpy(X_train)
X_test = convert_to_numpy(X_test)

# This is our first model which was a deep neural network that used rectified linear activation units
# Dropout was used as regularization of our network and each layer had 64 nodes
model = Sequential()
model.add(Dense(64, input_dim=25, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))

# [Mike] This functions compiles the above model with its associated loss function and optimizer
# We used mean squared error as the loss function with the Adam optimizer
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# This trains the model using 500 epochs and a batch size of 50
model.fit(X_train, y_train,
          epochs=500,
          batch_size=50, verbose=1)

# This was used to return the model's predictions (probabilities) on the test set
# As well as convert the probabilities to the class labels
y_pred_1 = model.predict(X_test)
y_pred_bool = (y_pred_1 >= 0.5)

# This evaluates the trained model on the test set
score, acc = model.evaluate(X_test, y_test, batch_size=100)
print('Test accuracy: ', acc)

cm = confusion_matrix(y_test, y_pred_bool)
print(cm)
print(roc_auc_score(y_test, y_pred_1))

# [Christina] This is our second model which was a shallow neural network that used hyperbolic tangent activation functions
# Dropout was used as regularization of our network and each layer had 64 nodes
model = Sequential()
model.add(Dense(64, input_dim=25, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='tanh'))

# This functions compiles the above model with its associated loss function and optimizer
# We used mean squared error as the loss function with the stochastic gradient descent optimizer
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['accuracy'])

# This trains the model using 500 epochs and a batch size of 50
model.fit(X_train, y_train,
          epochs=500,
          batch_size=50, verbose=1)

# This was used to return the model's predictions (probabilities) on the test set
# As well as convert the probabilities to the class labels
y_pred_2 = model.predict(X_test)
y_pred_bool = (y_pred_2 >= 0.5)

# This evaluates the trained model on the test sets
score, acc = model.evaluate(X_test, y_test, batch_size=100)
print('Test accuracy: ', acc)

cm = confusion_matrix(y_test, y_pred_bool)
print(cm)
print(roc_auc_score(y_test, y_pred_2))

# This is our third model which was a shallow neural network that used sigmoid activation functions
# Dropout was used as regularization of our network and each layer had 64 nodes
model = Sequential()
model.add(Dense(64, input_dim=25, activation='hard_sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='hard_sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='hard_sigmoid'))

# [Vinith] This functions compiles the above model with its associated loss function and optimizer
# We used binary cross entropy error as the loss function with the stochastic gradient descent optimizer
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# This trains the model using 500 epochs and a batch size of 50
model.fit(X_train, y_train,
          epochs=500,
          batch_size=50, verbose=1)

# This was used to return the model's predictions (probabilities) on the test set
# As well as convert the probabilities to the class labels
y_pred_3 = model.predict(X_test)
y_pred_bool = (y_pred_3 >= 0.5)

# This evaluates the trained model on the test sets
score, acc = model.evaluate(X_test, y_test, batch_size=100)
print('Test accuracy: ', acc)
cm = confusion_matrix(y_test, y_pred_bool)
print(cm)
print(roc_auc_score(y_test, y_pred_3))

# This section of code determines the false positive rates and true positive rates
# for each model in addition to the AUC so that the ROC Curves can be plotted
from sklearn.metrics import roc_curve
fpr_keras_1, tpr_keras_1, thresholds_keras_1 = roc_curve(y_test, y_pred_1.ravel())
from sklearn.metrics import auc
auc_keras_1 = auc(fpr_keras_1, tpr_keras_1)

fpr_keras_2, tpr_keras_2, thresholds_keras_2 = roc_curve(y_test, y_pred_2.ravel())
auc_keras_2 = auc(fpr_keras_2, tpr_keras_2)


fpr_keras_3, tpr_keras_3, thresholds_keras_3 = roc_curve(y_test, y_pred_3.ravel())
auc_keras_3 = auc(fpr_keras_3, tpr_keras_3)

# This code plots all of the ROC curves onto one plot so that they can be compared effectively
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras_1, tpr_keras_1, label='Model 1 (area = {:.3f})'.format(auc_keras_1))
plt.plot(fpr_keras_2, tpr_keras_2, label='Model 2 (area = {:.3f})'.format(auc_keras_2))
plt.plot(fpr_keras_3, tpr_keras_3, label='Model 3 (area = {:.3f})'.format(auc_keras_3))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
