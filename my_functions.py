# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:54:44 2018

@author: Philip
"""

import numpy as np



def get_data_poly_noise(start, stop, noise_rel=0.1, num=50, order=1, coeff=None):
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
    if coeff is not None:
        if len(coeff) != order+1:
            raise NameError('Number of coefficients must be equal to the order + 1!')
        else:
            poly_coeff = np.array(coeff)
    else:
        poly_coeff = 10 * np.random.random_sample(size=order+1) - 5
    
    #create polynomial
    y = np.zeros(x.shape)
    for i in range(order+1):
        y += poly_coeff[i] * x**i
    
    noise_mag = noise_rel * np.abs((np.max(y) - np.min(y)))
    #add noise in [-noise_mag/2, noise_mag/2]
    y += noise_mag * np.random.random_sample(size=num) - noise_mag/2
    
    return (x, y)