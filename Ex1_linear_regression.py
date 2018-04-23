#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:26:47 2018

@author: pmirbach
"""

import numpy as np
from matplotlib import pyplot as plt

from sklearn import linear_model



N_train = 200
N_test = 30




def get_data_poly_noise(start, stop, noise_rel=0.1, num=50, order=1):
    x = (stop - start) * np.random.random_sample(size=num) + start
#    x.reshape((-1,1))
    
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

x_train = x[:-N_test]
y_train = y[:-N_test]

x_test = x[-N_test:]


#machine learning stuff
reg = linear_model.LinearRegression()
reg.fit(x_train.reshape((-1,1)), y_train)

y_predict = reg.predict(x_test.reshape((-1,1)))






#Plotting:

fig, ax = plt.subplots()

axes = ax.plot(x_train, y_train, 'x', color='black', label='training data')
axes = ax.plot(x_test, y_predict, '-o', color='red', label='prediction')

ax.legend(loc='best')

fig.show()