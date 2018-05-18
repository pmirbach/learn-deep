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


#Loading data & preparing data
iris = datasets.load_iris()
X, y = iris.data, iris.target


#Machine learning Classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=1)


#Machine learning 
knn.fit(X,y)


#print(iris.target_names[knn.predict([[3,5,4,2]])])





def plot_lims(x, surp_perc=0.08):
    x_lower = x.min() - np.abs(x.max() - x.min()) * surp_perc
    x_upper = x.max() + np.abs(x.max() - x.min()) * surp_perc
    return (x_lower, x_upper)



def plot_cls(data, target, target_names=None, feature_names=None,
             plot_dim=[0,1], classifier=None):
    
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
    
    
    if classifier is not None:
        
        x_lims = plot_lims(data[:,x_ind])
        y_lims = plot_lims(data[:,y_ind])
        
        ax.set(xlim=x_lims, ylim=y_lims)
        
        X, Y = np.meshgrid(np.linspace(x_lims[0], x_lims[1]),
                           np.linspace(x_lims[0], x_lims[1]))
        
        x_pred = np.c_[X.ravel(), Y.ravel()]
        print(x_pred.shape)
        
        
         Z = classifier.predict(x_pred)
#        Z.reshape(X.shape)
#        
#        plt.pcolormesh(X, Y, Z)

        
        
        
        
    
    
    
    return fig

fig1 = plot_cls(iris.data, iris.target, target_names=iris.target_names,
                feature_names=iris.feature_names, plot_dim=[0,1],
                classifier=knn)











