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



n_layers = 4



X, y = datasets.make_classification(n_samples=5000, n_features=10, n_informative=2)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

clf_NN = neural_network.MLPClassifier(hidden_layer_sizes=(100,)*n_layers, activation='logistic', max_iter=500)

clf_NN.fit(X_train, y_train)
y_predict = clf_NN.predict(X_test)


NN_shape = [coef.shape for coef in clf_NN.coefs_]

print(NN_shape)
print(np.sum(np.abs(y_test - y_predict)) / y_test.size)


fig, axes = plt.subplots(n_layers,1)
ims = []

for i in range(n_layers):
    im = axes[i].imshow(clf_NN.coefs_[i])
#    fig.colorbar(im)
    ims.append(im)

#print(len(axes))

#
#im = ax.imshow(clf_NN.coefs_[0])
#cb = fig.colorbar(ims[0])
#
plt.show()




#fig, ax = plt.subplots()
#ax.plot(y_test - y_predict)
##ax.plot(y_predict)
#
#plt.show()
