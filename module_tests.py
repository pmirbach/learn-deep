#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 18:53:11 2018

@author: pmirbach
"""

#import numpy as np
#import matplotlib.pyplot as plt


#
#from sklearn import svm
#from sklearn import datasets
#iris = datasets.load_iris()
#digits = datasets.load_digits()
#
#clf = svm.SVC(gamma=0.001, C=100.)
#clf.fit(digits.data[:-1], digits.target[:-1])
#print(clf.predict(digits.data[-1:]))


#import tensorflow as tf
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))




#import torch
#
#dtype = torch.float
#device = torch.device("cpu")
## dtype = torch.device("cuda:0") # Uncomment this to run on GPU
#
## N is batch size; D_in is input dimension;
## H is hidden dimension; D_out is output dimension.
#N, D_in, H, D_out = 64, 1000, 100, 10
#
## Create random input and output data
#x = torch.randn(N, D_in, device=device, dtype=dtype)
#y = torch.randn(N, D_out, device=device, dtype=dtype)
#
## Randomly initialize weights
#w1 = torch.randn(D_in, H, device=device, dtype=dtype)
#w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#
#learning_rate = 1e-6
#for t in range(500):
#    # Forward pass: compute predicted y
#    h = x.mm(w1)
#    h_relu = h.clamp(min=0)
#    y_pred = h_relu.mm(w2)
#
#    # Compute and print loss
#    loss = (y_pred - y).pow(2).sum().item()
#    print(t, loss)
#
#    # Backprop to compute gradients of w1 and w2 with respect to loss
#    grad_y_pred = 2.0 * (y_pred - y)
#    grad_w2 = h_relu.t().mm(grad_y_pred)
#    grad_h_relu = grad_y_pred.mm(w2.t())
#    grad_h = grad_h_relu.clone()
#    grad_h[h < 0] = 0
#    grad_w1 = x.t().mm(grad_h)
#
#    # Update weights using gradient descent
#    w1 -= learning_rate * grad_w1
#    w2 -= learning_rate * grad_w2



#from keras.models import Sequential
#
#model = Sequential()
#
#from keras.layers import Dense
#
#model.add(Dense(units=64, activation='relu', input_dim=100))
#model.add(Dense(units=10, activation='softmax'))
#
#model.compile(loss='categorical_crossentropy',
#              optimizer='sgd',
#              metrics=['accuracy'])



#from __future__ import print_function
#
#import numpy as np
#import tflearn
#
## Download the Titanic dataset
#from tflearn.datasets import titanic
#titanic.download_dataset('titanic_dataset.csv')
#
## Load CSV file, indicate that the first column represents labels
#from tflearn.data_utils import load_csv
#data, labels = load_csv('titanic_dataset.csv', target_column=0,
#                        categorical_labels=True, n_classes=2)
#
#
## Preprocessing function
#def preprocess(data, columns_to_ignore):
#    # Sort by descending id and delete columns
#    for id in sorted(columns_to_ignore, reverse=True):
#        [r.pop(id) for r in data]
#    for i in range(len(data)):
#      # Converting 'sex' field to float (id is 1 after removing labels column)
#      data[i][1] = 1. if data[i][1] == 'female' else 0.
#    return np.array(data, dtype=np.float32)
#
## Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
#to_ignore=[1, 6]
#
## Preprocess data
#data = preprocess(data, to_ignore)
#
## Build neural network
#net = tflearn.input_data(shape=[None, 6])
#net = tflearn.fully_connected(net, 32)
#net = tflearn.fully_connected(net, 32)
#net = tflearn.fully_connected(net, 2, activation='softmax')
#net = tflearn.regression(net)
#
## Define model
#model = tflearn.DNN(net)
## Start training (apply gradient descent algorithm)
#model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)
#
## Let's create some data for DiCaprio and Winslet
#dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
#winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
## Preprocess data
#dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
## Predict surviving chances (class 1 results)
#pred = model.predict([dicaprio, winslet])
#print("DiCaprio Surviving Rate:", pred[0][1])
#print("Winslet Surviving Rate:", pred[1][1])





















