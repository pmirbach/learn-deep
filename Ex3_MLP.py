# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:09:19 2018

@author: Philip
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn import model_selection
from sklearn import neural_network


#statistical evaluation
flg_stat = 1



n_layers = 4



X, y = datasets.make_classification(n_samples=5000, n_features=10, n_informative=2)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)



res = np.zeros((10,6))

out_str = 'Percentage of false predictions - {} hidden layers'

for j in range(6):
    for i in range(10):
        clf_NN = neural_network.MLPClassifier(hidden_layer_sizes=(70,)*j, activation='logistic', max_iter=500)
        clf_NN.fit(X_train, y_train)
        y_predict = clf_NN.predict(X_test)
        res[i,j] = np.sum(np.abs(y_test - y_predict)) / y_test.size * 100
        
    print(out_str.format(j+1))
    print(np.mean(res[:,j]))
    
    #    print(np.sum(np.abs(y_test - y_predict)) / y_test.size)

#print(res)



#clf_NN = neural_network.MLPClassifier(hidden_layer_sizes=(100,)*n_layers, activation='logistic', max_iter=500)
#
#clf_NN.fit(X_train, y_train)
#y_predict = clf_NN.predict(X_test)


#NN_shape = [coef.shape for coef in clf_NN.coefs_]

#print(NN_shape)
#print(np.sum(np.abs(y_test - y_predict)) / y_test.size)


#fig, axes = plt.subplots(n_layers,1)
#ims = []
#
#for i in range(n_layers):
#    im = axes[i].imshow(clf_NN.coefs_[i])
##    fig.colorbar(im)
#    ims.append(im)
#
##print(len(axes))

#
#im = ax.imshow(clf_NN.coefs_[0])
#cb = fig.colorbar(ims[0])
#
#plt.show()




#fig, ax = plt.subplots()
#ax.plot(y_test - y_predict)
##ax.plot(y_predict)
#
#plt.show()
