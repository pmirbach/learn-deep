#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 16:42:44 2018

@author: pmirbach
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import neighbors


K = 5                   #Number of considered neighbors
#Method for the weight of the data.
w_method = 'uniform'
#w_method = 'distance'
poly_order = 1          #Order of the polynomial
N_train = 200           #number training points


def get_data_poly_noise(start, stop, noise_rel=0.1, num=50, order=1):
    """Return noisy data based on a polynomail function.
    
    The domain of the function is randomly generated, as well as the coefficients of the polynomial.
    
    Parameters
    ---------
    start: scalar
        The starting value of the domain.
    stop: scalar
        The end value of the domain.
    noise_rel: scalar, optional
        The noise added to the data relative to the range of the values. Default is 0.1.
    num: int, optional
        Number of generated sample points. Default is 50.
    order: int, optional
        The order of the polynomial function. Coefficients are randomly generated in range [-5,5]. 
        Default is a linear funtion (order=1)
    
    Returns
    -------
    domain: ndarray
        The randomly generated domain x.
    values: ndarray
        The corresponding values of the polynomial with added noise.
    """
    
    x = (stop - start) * np.random.random_sample(size=num) + start    
    #coefficients for the polynomial in [-5,5]
    poly_coeff = 10 * np.random.random_sample(size=order+1) - 5
    
    #create polynomial
    y = np.zeros(x.shape)
    for i in range(order+1):
        y += poly_coeff[i] * x**i
    
    noise_mag = noise_rel * np.abs((np.max(y) - np.min(y)))
    #add noise in [-noise_mag/2, noise_mag/2]
    y += noise_mag * np.random.random_sample(size=num) - noise_mag/2
    
    return (x, y)


#create training data
(x_train, y_train) = get_data_poly_noise(start=-20, stop=30, num=N_train, noise_rel=0.2, order=poly_order)

#create testing data - try to cover full area to see predictions between training data points (e.g. plateaus for K=1)
x_test = np.linspace(start=-20, stop=30, num=1000)



### machine learning stuff ###
neigh = neighbors.KNeighborsRegressor(n_neighbors=K, weights=w_method)
neigh.fit(x_train.reshape((-1,1)), y_train)
y_predict = neigh.predict(x_test.reshape((-1,1)))


#Plotting:
fig, ax = plt.subplots()

axes = ax.plot(x_train, y_train, 'x', color='black', label='training data')
axes = ax.plot(x_test, y_predict, '-', color='red', label='prediction')

ax.legend(loc='best')

fig.show()