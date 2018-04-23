#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:42:44 2018

@author: pmirbach
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import neighbors


K = 5                   #Number of neighbors
w_method = 'uniform'
#w_method = 'distance'
poly_order = 1          #Order of the polynomial
N_train = 200           #number training points


def get_data_poly_noise(start, stop, num=50, noise_rel=0.1, order=1):
    x = (stop - start) * np.random.random_sample(size=num) + start
    
    #coefficients for the polynomial in [-5,5]
    poly_coeff = 10 * np.random.random_sample(size=order+1) - 5
    
    #create polynomial
    y = np.zeros(x.shape)
    for i in range(order+1):
        y += poly_coeff[i] * x**i
    
    #create noise relative to data
    noise_mag = noise_rel * np.abs((np.max(y) - np.min(y)))
    #add noise in [-noise_mag/2, noise_mag/2]
    y += noise_mag * np.random.random_sample(size=num) - noise_mag/2
    
    return (x, y)


#create data
(x_train, y_train) = get_data_poly_noise(start=-20, stop=30, num=N_train, noise_rel=0.2, order=poly_order)


x_test = np.linspace(start=-20, stop=30, num=1000)





#machine learning stuff
neigh = neighbors.KNeighborsRegressor(n_neighbors=K, weights=w_method)
neigh.fit(x_train.reshape((-1,1)), y_train)
y_predict = neigh.predict(x_test.reshape((-1,1)))


#neighbors.KNeighborsRegressor()



#Plotting:

fig, ax = plt.subplots()

axes = ax.plot(x_train, y_train, 'x', color='black', label='training data')
axes = ax.plot(x_test, y_predict, '-', color='red', label='prediction')

ax.legend(loc='best')

fig.show()