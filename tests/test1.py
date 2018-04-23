#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:21:11 2018

@author: pmirbach
"""

import numpy as np



x = np.linspace(start=0, stop=50, num=50)
#standard 1D shape: (N,)
print(x.shape)

#1D data in correct shape: (N,1)
x_new = x.reshape((-1,1))
print(x_new.shape)