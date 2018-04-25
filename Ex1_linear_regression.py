#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:26:47 2018

@author: pmirbach
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model


#Number training / testing data
N_train = 200
N_test = 30


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


#create data
(x, y) = get_data_poly_noise(start=-20, stop=30, noise_rel=0.2, num=(N_train + N_test), order=1)

#split data in training and testing data
x_train = x[:-N_test]
y_train = y[:-N_test]

x_test = x[-N_test:]


### machine learning stuff ###
reg = linear_model.LinearRegression()
reg.fit(x_train.reshape((-1,1)), y_train)

y_predict = reg.predict(x_test.reshape((-1,1)))




#Plotting:

fig, ax = plt.subplots()

axes = ax.plot(x_train, y_train, 'x', color='black', label='training data')
axes = ax.plot(x_test, y_predict, '-o', color='red', label='prediction')

ax.legend(loc='best')

fig.show()