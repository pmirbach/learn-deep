# -*- coding: utf-8 -*-
"""
Created on Fri May 25 08:49:14 2018

@author: Philip
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network, model_selection
from my_functions import get_data_poly_noise


N_class_0 = 1000
N_class_1 = 1000



#Creation of well seperated U-shape data
#U with parable
x1, y1 = get_data_poly_noise(start=-10, stop=10, noise_rel=0.2, num=N_class_0, order=2, coeff=[0,0,1])
x2, y2 = get_data_poly_noise(start=-5, stop=5, noise_rel=0.2, num=N_class_1, order=2, coeff=[50,0,1])


X1 = np.column_stack((x1, y1))
X2 = np.column_stack((x2, y2))

X = np.concatenate((X1, X2), axis=0)
y = np.concatenate((np.zeros((N_class_0,)), np.ones((N_class_1,))), axis=0)

clf_NN = neural_network.MLPClassifier(
        hidden_layer_sizes=(3,), activation='logistic', max_iter=200)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)


clf_NN.fit(X_train, y_train)
y_predict = clf_NN.predict(X_test)


out_str_NN = 'Percentage of false predictions - Neural network: {}'
err_NN = np.sum(np.abs(y_test - y_predict)) / y_test.size

print(out_str_NN.format(err_NN))



#fig, ax = plt.subplots()
#ax.plot(x1, y1, 'x')
#ax.plot(x2, y2, 'x', 'red')
#
#plt.show()




