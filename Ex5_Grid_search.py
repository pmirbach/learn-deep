# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:02:02 2018

@author: Philip
"""


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#from sklearn.metrics import 
#import statsmodels.api as sm


X,y = datasets.load_digits(return_X_y=True)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)



tuned_parameters = [{'solver': ['adam', 'sgd']}, 
                     {'hidden_layer_sizes': 
                         [(50,50),(50,100),(100,50),(100,100)]}]


#if __name__ == "__main__":
#    X_train, y_train, X_test, y_test = load_mnist(one_hot=True)
#    y_pred = mlp1(X_train, y_train, X_test, y_test)

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    




