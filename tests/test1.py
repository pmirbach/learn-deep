#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:21:11 2018

@author: pmirbach
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

from sklearn import datasets
from sklearn import model_selection


###correc shape for 1d data

#x = np.linspace(start=0, stop=50, num=50)
##standard 1D shape: (N,)
#print(x.shape)
#
##1D data in correct shape: (N,1)
#x_new = x.reshape((-1,1))
#print(x_new.shape)


###data sets 
#boston = datasets.load_boston()
#print(boston.DESCR)


##X, y = datasets.make_blobs(n_samples=1000, centers=3, n_features=2)
#X, y = datasets.make_classification(n_samples=1000, n_features=2, 
#                                    n_informative=2, n_redundant=0,
#                                    n_clusters_per_class=1, n_classes=3)
#
#y_unique = np.unique(y)
#colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
#
#fig1, ax = plt.subplots()
#for (this_y, color) in zip(y_unique, colors):
#    this_X = X[y == this_y]
#    ax.scatter(this_X[:,0], this_X[:,1], c=color)
#
##model_selection.train_test_split()
#
##exit()
#
#
#plt.show()
#
##model_selection.train_test_split()

#print((1,)*2)


#a = [1,2,3]
#b = [4,5,6]
#
#dic_1 = dict(zip(a,b))
#print(dic_1[1])




a = 1

print(type(a))

















