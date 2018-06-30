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
# Chapter 7: convolutional neural network
#-----------------------------------------------------------------

def conv(X_train, y_train, X_test, y_test):

	# reshape data
	X_train = X_train.reshape([-1, 28, 28, 1])
	X_test = X_test.reshape([-1, 28, 28, 1])

	# build network
	network = tflearn.input_data(shape=[None, 28, 28, 1], name='input')
	network = tflearn.conv_2d(network, 32, 3, activation='relu', regularizer="L2")
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.local_response_normalization(network)
	network = tflearn.conv_2d(network, 64, 3, activation='relu', regularizer="L2")
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.local_response_normalization(network)
	network = tflearn.fully_connected(network, 128, activation='tanh')
	network = tflearn.dropout(network, 0.8)
	network = tflearn.fully_connected(network, 256, activation='tanh')
	network = tflearn.dropout(network, 0.8)
	network = tflearn.fully_connected(network, 10, activation='softmax')
	network = tflearn.regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

	# train conv net
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.fit({'input': X_train}, {'target': y_train}, n_epoch=20, validation_set=({'input': X_test}, {'target': y_test}), snapshot_step=100, show_metric=True)

#-----------------------------------------------------------------
# Chapter 8: neuroevo
#-----------------------------------------------------------------

	# hyperparameters
units = [2**(units_+3) for units_ in range(2**3)] # 3 bits
activations = ['tanh','sigmoid','relu','Softsign'] # 2 bits
values = [0.25,0.5,0.75,0.9] # 2 bits

	# from binary to int
def toInt(z):
	z=z[::-1]
	sum_=0
	for i in range(len(z)):
		sum_+=2**i*z[i]
	return sum_

	# index and type handling
def unwrap(genotype,z):
	global k, units, activations, values
	print ("k",k)
	if genotype[0] == 8:
		k+=3
		return units[toInt(z[k-3:k])]
	elif genotype[0] == 'tanh':
		k+=2
		return activations[toInt(z[k-2:k])]
	elif genotype[0] == 0.25:
		k+=2
		return values[toInt(z[k-2:k])]

	# fitness function
def fit(z, X_train, y_train, X_test, y_test):

	# resphape data
	X_train = np.reshape(X_train, (-1, 28, 28))
	X_test = np.reshape(X_test, (-1, 28, 28))

	global k, units, activations, values
	k = 0

	# conv net
	tf.reset_default_graph()
	network = tflearn.input_data(shape=[None, 28, 28, 1], name='input')
	network = tflearn.conv_2d(network, unwrap(units,z), 3, activation=unwrap(activations,z), regularizer="L2")
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.local_response_normalization(network)
	network = tflearn.conv_2d(network, unwrap(units,z), 3, activation=unwrap(activations,z), regularizer="L2")
	network = tflearn.max_pool_2d(network, 2)
	network = tflearn.local_response_normalization(network)
	network = tflearn.fully_connected(network, unwrap(units,z), activation=unwrap(activations,z))
	network = tflearn.dropout(network, unwrap(values,z))
	network = tflearn.fully_connected(network, unwrap(units,z), activation=unwrap(activations,z))
	network = tflearn.dropout(network, unwrap(values,z))
	network = tflearn.fully_connected(network, 10, activation='softmax')
	network = tflearn.regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

	# training
	model = tflearn.DNN(network, tensorboard_verbose=0)
	model.fit({'input': X_train}, {'target': y_train}, n_epoch=20, validation_set=({'input': X_test}, {'target': y_test}), snapshot_step=100, show_metric=True)
	score = model.evaluate(testX, testY)
	print ("acc = ", score[0])
	return score[0]

	# GA
def ea(X_train, y_train, X_test, y_test):

	# init
	n = 24
	sigma = 1./n
	z = np.random.randint(2, size=(n,))
	f = fit(z,X_train, y_train, X_test, y_test)
	output = "initialization: fitness: "+str(f)+"\n"
	print (output)

	# GA loop
	for t in range(100):

		child = [(bit+1)%2 if np.random.random()<sigma else bit for bit in z]
		f_ = fit(child, X_train, y_train, X_test, y_test)
		if -f_<-f:
			z = child
			f = f_
	
		output = "generation: "+str(t)+", fitness: "+str(f)+"\n"
		print (output)
	output = "fitness of: "+str(z)+" is: "+str(f)
	print (output)

#-----------------------------------------------------------------
# Chapter 9: autoencoder
#-----------------------------------------------------------------

def auto(X_train, y_train, X_test, y_test):

	# build network
	encoder = tflearn.input_data(shape=[None, 784])
	encoder = tflearn.fully_connected(encoder, 256)
	encoder = tflearn.fully_connected(encoder, 64)

	decoder = tflearn.fully_connected(encoder, 256)
	decoder = tflearn.fully_connected(decoder, 784, activation='sigmoid')

	# regression
	network = tflearn.regression(decoder, optimizer='adam', learning_rate=0.001, loss='mean_square', metric=None)
	
	# train autoencoder
	model = tflearn.DNN(network, tensorboard_verbose=0)
   #print(X_train.shape)
	#raise('STOP')
	X_train_noise = X_train + np.random.randn(X_train.shape)
	raise('STOP')
	
	model.fit(X_train, X_train, n_epoch=20, validation_set=(X_test, X_test), run_id="auto_encoder", batch_size=256)

	# encode X_train[0] for test
	encoding_model = tflearn.DNN(encoder, session=model.session)
	print(encoding_model.predict([X_train[0]]))

	# testing image reconstruction on new data
	print("Visualizing results after being encoded and decoded:")
	X_test = tflearn.data_utils.shuffle(X_test)[0]
	# apply encode and decode over test set
	encode_decode = model.predict(X_test)
	# compare original images with their reconstructions
	f, a = plt.subplots(2, 10, figsize=(10, 2))
	for i in range(10):
	    temp = [[ii, ii, ii] for ii in list(X_test[i])]
	    a[0][i].imshow(np.reshape(temp, (28, 28, 3)))
	    temp = [[ii, ii, ii] for ii in list(encode_decode[i])]
	    a[1][i].imshow(np.reshape(temp, (28, 28, 3)))
	f.show()
	plt.draw()
	plt.waitforbuttonpress()


#-----------------------------------------------------------------
# run
#-----------------------------------------------------------------

#X_train, y_train, X_test, y_test = load_mnist(one_hot=False)
#print ("kNN outputs",knn_short(X_test[0], X_train, y_train, k=10))
#y_pred = mlp1(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = load_mnist(one_hot=True)
#y_pred = mlp2(X_train, y_train, X_test, y_test)
#grid(X_train, y_train, X_test, y_test)
#report(y_test, y_pred)
#conv(X_train, y_train, X_test, y_test)
#ea(X_train, y_train, X_test, y_test)
auto(X_train, y_train, X_test, y_test)
#gan(X_train, y_train, X_test, y_test)
plt.show()