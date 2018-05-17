# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:37:16 2018

@author: Philip
"""

#import sklearn
from sklearn import neighbors, datasets

import numpy as np
import matplotlib.pyplot as plt


#Flags
flg_plot = 0


#Loading data & preparing data
iris = datasets.load_iris()
X, y = iris.data, iris.target


#Machine learning Classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=1)


#Machine learning 
knn.fit(X,y)


#print(iris.target_names[knn.predict([[3,5,4,2]])])






if flg_plot:
    ### Nice scatter plot
    x_index = 0
    y_index = 1
    
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
    
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
    fig.colorbar(im, ticks=np.unique(iris.target), format=formatter)
#    fig.colorbar(im, ticks=[0,1,2], format=None)
    
    ax.set(xlabel=iris.feature_names[x_index], 
           ylabel=iris.feature_names[y_index])



def plot_cls(data, target, target_names=None, feature_names=None,
             plot_dim=[0,1], cls=None):
    
    x_ind = plot_dim[0]
    y_ind = plot_dim[1]
    
    if target_names is not None:
        formatter = plt.FuncFormatter(lambda i, *args: target_names[i])
    else:
        formatter = None
    
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.scatter(data[:,x_ind], data[:,y_ind], c=target)
    fig.colorbar(im, ticks=np.unique(target), format=formatter)
    
    if feature_names is not None:
        ax.set(xlabel=iris.feature_names[x_ind], 
               ylabel=iris.feature_names[y_ind])
    
    if cls is not None:
        pass
    
    return fig

fig1 = plot_cls(iris.data, iris.target, target_names=iris.target_names,
                feature_names=iris.feature_names, plot_dim=[0,1])











