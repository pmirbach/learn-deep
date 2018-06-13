# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 08:59:55 2018

@author: Philip
"""


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import batch_normalization

from tflearn.datasets import mnist


import tensorflow
tensorflow.reset_default_graph()



flg_train = 1


def load_mnist(one_hot=True):
    X_train, y_train, X_test, y_test = mnist.load_data(one_hot=one_hot)
    X_train = X_train.reshape([-1, 28, 28, 1])
    X_test = X_test.reshape([-1, 28, 28, 1])
    return X_train, y_train, X_test, y_test

    
# build network
network = input_data(shape=[None, 28, 28, 1])
network = conv_2d(network, nb_filter=4, filter_size=5, activation='relu', regularizer='L2')
network = max_pool_2d(network, kernel_size=2)
network = batch_normalization(network)

network = conv_2d(network, nb_filter=4, filter_size=5, activation='relu', regularizer='L2')
network = max_pool_2d(network, kernel_size=2)
network = batch_normalization(network)

network = fully_connected(network, n_units=128, activation='relu', regularizer='L2', weight_decay=0.001)
network = dropout(network, 0.7)

network = fully_connected(network, n_units=256, activation='relu', regularizer='L2', weight_decay=0.001)
network = dropout(network, 0.7)

network = fully_connected(network, n_units=10, activation='softmax')
    
    

# network optimization
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=500)
top_k = tflearn.metrics.Top_k(3)
network = tflearn.regression(network, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

model = tflearn.DNN(network, tensorboard_verbose=0)



X_train, y_train, X_test, y_test = load_mnist(one_hot=True)


# train MLP
if flg_train:
    model.fit(X_train, y_train, n_epoch=20, validation_set=(X_test, y_test),
          show_metric=False, run_id="conv_model")
else:
    model.load("model.tfl")
            

# predict
y_pred = model.predict(X_test)








