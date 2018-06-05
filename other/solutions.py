#-----------------------------------------------------------------
# Python / scikit-learn / TFLearn scripts for exercises
# of "Deep Learning Essentials" by Oliver Kramer
#
# Scripts are partially based on scikit-learn and TFLearn examples
#-----------------------------------------------------------------

#-----------------------------------------------------------------
# Chapter 1: load data
#-----------------------------------------------------------------

import tflearn

def load_mnist(one_hot=True):
	X_train, y_train, X_test, y_test = tflearn.datasets.mnist.load_data(one_hot=one_hot)
	return X_train, y_train, X_test, y_test

#-----------------------------------------------------------------
# Chapter 2: kNN
#-----------------------------------------------------------------

import numpy as np
import scipy.spatial.distance

def knn_short(x, x_train, y_train, k=10):
	# kNN in one line
	return np.bincount([label for _,label in sorted(zip([scipy.spatial.distance.euclidean(x, x_) for x_ in x_train],y_train))][:k]).argmax()


#-----------------------------------------------------------------
# Chapter 3: MLP in scikit-learn
#-----------------------------------------------------------------

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def mlp1(X_train, y_train, X_test, y_test):

	# train MLP
	mlp = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=100, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=0.1)
	mlp.fit(X_train, y_train)
	y_pred = mlp.predict(X_test)

	# scores
	print("Training set score: %f" % mlp.score(X_train, y_train))
	print("Test set score: %f" % mlp.score(X_test, y_test))

	# print weights
	fig, axes = plt.subplots(4, 4)
	vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
	for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
		ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
	ax.set_xticks(())
	ax.set_yticks(())
	return y_pred

#-----------------------------------------------------------------
# Chapter 4: MLP in TFLearn
#-----------------------------------------------------------------

def mlp2(X_train, y_train, X_test, y_test):
	
	# build network
	network = tflearn.input_data(shape=[None, 784])
	network = tflearn.fully_connected(network, 128, activation='tanh', regularizer='L2', weight_decay=0.001)
	network = tflearn.dropout(network, 0.7)
	network = tflearn.fully_connected(network, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
	network = tflearn.dropout(network, 0.7)
	network = tflearn.fully_connected(network, 10, activation='softmax')

	# network optimization
	sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
	top_k = tflearn.metrics.Top_k(3)
	network = tflearn.regression(network, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

	# train MLP
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.fit(X_train, y_train, n_epoch=20, validation_set=(X_test, y_test),
	          show_metric=True, run_id="dense_model")

	# predict
	y_pred = model.predict(X_test)
	return y_pred

#-----------------------------------------------------------------
# Chapter 5: cross-validation and grid search
#-----------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def grid(X_train, y_train, X_test, y_test):

	# grid search solution space
	tuned_parameters = [{'hidden_layer_sizes': [(50,50),(50,100),(100,50),(100,100)], 'solver': ['sgd','adam']}]
	
	# scores to consider
	scores = ['precision', 'recall']

	for score in scores:

	# grid search
	    clf = GridSearchCV(MLPClassifier(verbose=10, max_iter = 20), tuned_parameters, cv=5, scoring='%s_macro' % score)
	    clf.fit(X_train, y_train)
	    print("Best parameters:",clf.best_params_)
	    
	# scores
	    print("Grid scores:")
	    means = clf.cv_results_['mean_test_score']
	    stds = clf.cv_results_['std_test_score']
	    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
	    y_true, y_pred = y_test, clf.predict(X_test)

#-----------------------------------------------------------------
# Chapter 6: classification report and confusion matrix
#-----------------------------------------------------------------

def report(y_true, y_pred):

	# classification report
	print("Classification report for classifier:\n%s\n" % (metrics.classification_report(y_true, y_pred)))

	# confusion matrix
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_true, y_pred))



#-----------------------------------------------------------------
# run
#-----------------------------------------------------------------

X_train, y_train, X_test, y_test = load_mnist(one_hot=False)
print ("kNN outputs",knn_short(X_test[0], X_train, y_train, k=10))
y_pred = mlp1(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = load_mnist(one_hot=True)
y_pred = mlp2(X_train, y_train, X_test, y_test)
grid(X_train, y_train, X_test, y_test)
report(y_test, y_pred)