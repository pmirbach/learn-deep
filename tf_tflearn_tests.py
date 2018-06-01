# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 08:09:59 2018

@author: Philip
"""

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))