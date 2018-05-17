from sklearn import datasets
import numpy as np
import scipy
iris = datasets.load_iris()


def knn(x, x_train, y_train, k=10):

	ground_labels = np.unique(y_train)
	D = []
	for x_ in x_train:
		d = scipy.spatial.distance.euclidean(x, x_)
		D.append(d)
	liste = zip(D,y_train)
	sorted_labels = [label for _,label in sorted(liste)][:k]

	y = np.bincount(sorted_labels).argmax()
	print ("unser label lautet",y)

	return y


def knn_short(x, x_train, y_train, k=10):
	return np.bincount([label for _,label 
                     in sorted(zip([scipy.spatial.distance.euclidean(x, x_) 
                     for x_ in x_train],y_train))][:k]).argmax()


x = iris.data[80]
y = knn(x,iris.data,iris.target)
print (y)

y = knn_short(x,iris.data,iris.target)
print (y)



