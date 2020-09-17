#!/usr/bin/env python
# coding: utf-8


# to see if Script runs in my created conda environment
import sys

sys.executable

# import important packages
import pandas as pd
import numpy as np
import pm4py
import tensorflow as tf
import matplotlib.pyplot as plt

# load log data and save it in variable log
from pm4py.objects.log.importer.xes import importer as xes_importer

log = xes_importer.apply("C:\\Users\\kleeb\\Master\\Data Analytics Seminar\\InternationalDeclarations.xes.gz")

# transform the pm4py data into a pandas dataframe and display it
from pm4py.objects.conversion.log import converter as log_converter

dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
display(dataframe)

# By taking a summarizing look at the data set it seems that the events are already chronologically ordered
# However, the following code makes sure that events are indeed in chronological order for each case

# d1 is a list of ascendingly ordered timestamps for one particular case
# d2 is a list of timestamps for one case as it was available
# in each iteration timestamps of a new case are assigned
# Note index must be dropped - If not, check would always return False

cases = dataframe['case:concept:name'].unique()

# first transform dates from strings to datetime objects to order them with sort_values()
dataframe['time:timestamp'] = pd.to_datetime(dataframe['time:timestamp'], utc=True)

for i in range(len(cases)):

    d1 = dataframe[dataframe['case:concept:name'] == cases[i]].sort_values(by='time:timestamp', ascending=
    True)['time:timestamp'].reset_index(drop=True)
    d2 = dataframe[dataframe['case:concept:name'] == cases[i]]['time:timestamp'].reset_index(drop=True)
    x = d1 != d2

    for j in range(len(x)):
        if x[j] == True:
            print('The Data is not perfectly chronologically ordered')

# create Output variable y and rename the output column to Output
y = dataframe[['case:concept:name', 'concept:name']]
y = y.shift(-1)
y.rename(columns={'concept:name': 'Output'}, inplace=True)

# create input variable matrix x
x = dataframe[['case:concept:name', 'concept:name']]

# create dataset z containing input and output variable
z = x.merge(y, right_index=True, left_index=True)
# checking whether case id matches for input and output and save that boolean list in the variable mask
# idea: find values in data set where the case id of our input variables does not match the case id of the output variable.
# In such a case, the output variable is set to the value 'Last event' since this is the last event in a case and
# there is no event left to predict by the Neural Net
z['mask'] = (z['case:concept:name_x'] == z['case:concept:name_y'])
z.loc[z['mask'] == False, 'Output'] = 'Last event'

# Strings cannot be used in NN. Thus, one hot encode X and integer encode y variable
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X = pd.DataFrame(z['concept:name'])
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoder.fit(X)

X_one_encoded = one_hot_encoder.transform(X)
X_one_encoded = pd.DataFrame(data=X_one_encoded, columns=one_hot_encoder.categories_)

encoder = LabelEncoder()
# Apply the label encoder to column input variable named
y_encoded = z[['Output']].apply(encoder.fit_transform)
z['Output_encoded'] = y_encoded

# create train/test split - 80% Training 20% testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_one_encoded, y_encoded, test_size=0.2, random_state=42)

# creating lists of evaluation metrics
accuracy = {}
weighted_precision = {}
weighted_recall = {}
weighted_f_measure = {}
macro_precision = {}
macro_recall = {}
macro_f_measure = {}
metrics = [accuracy,
           weighted_precision,
           weighted_recall,
           weighted_f_measure,
           macro_precision,
           macro_recall,
           macro_f_measure
           ]

# creating multiple ANNs
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report

hidden_layers = [1, 2, 3]
nodes = [10, 20, 30, 40]
epochs = [3, 5, 10]

for layer in hidden_layers:
    for node in nodes:
        for epoch in epochs:

            NAME = '{}-nodes-{}-hiddenlayers-{}-epochs'.format(node, layer, epoch)
            print(NAME)
            # (Sequential is feedforward NN)
            # The model expects rows of data with 34 variables (the input_dim=34 argument)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(34, input_dim=34, activation='relu'))
            # hidden layers
            for l in range(layer):
                model.add(tf.keras.layers.Dense(node, activation='relu'))
            # Outpt layer with 35 nodes (integer representing the next event)
            model.add(tf.keras.layers.Dense(35, activation=tf.nn.softmax))

            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test), callbacks=[tensorboard])

            # evaluation metrics
            y_pred = model.predict(X_test)
            y_pred_bool = np.argmax(y_pred, axis=1)
            model_metrics = classification_report(y_test, y_pred_bool, output_dict=True)
            accuracy[NAME] = model_metrics['accuracy']
            weighted_precision[NAME] = model_metrics['weighted avg']['precision']
            weighted_recall[NAME] = model_metrics['weighted avg']['recall']
            weighted_f_measure[NAME] = model_metrics['weighted avg']['f1-score']
            macro_precision[NAME] = model_metrics['macro avg']['precision']
            macro_recall[NAME] = model_metrics['macro avg']['recall']
            macro_f_measure[NAME] = model_metrics['macro avg']['f1-score']

Evaluation_data = pd.DataFrame(metrics, index=['accuracy', 'weighted_precision', 'weighted_recall',
                                               'weighted_f_measure', 'macro_avg_precision',
                                               'macro_avg_recall', 'macro_avg_f_measure']).transpose()
Evaluation_data.to_excel('Evaluation_data_2_Abgabe.xlsx')

# analyzing predictions of not prevalant classes via classification report
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(34, input_dim=34, activation='relu'))
model.add(tf.keras.layers.Dense(20, activation='relu'))
model.add(tf.keras.layers.Dense(35, activation=tf.nn.softmax))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred_bool = np.argmax(y_pred, axis=1)
metrics = classification_report(y_test, y_pred_bool, output_dict=True)
metrics = pd.DataFrame(metrics).transpose()
metrics.to_excel('Evaluation_metrics_test_set_best.xlsx')
metrics

# Encoded output next to corresponding acttivity
output_enc_nenc = pd.DataFrame({'Output_encoded': z['Output_encoded'].unique(), 'Output': z['Output'].unique()})
output_enc_nenc = output_enc_nenc.sort_values(by='Output_encoded', ascending=True)
output_enc_nenc.to_excel('Output_integer_encoded.xlsx')
output_enc_nenc






