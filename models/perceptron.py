import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from helpers.compute_score import print_confusion_matrix

from sklearn.linear_model import Perceptron

def perceptron(x_train, x_test, y_test, y_train):
    print("\n\t >> Fit in Model:")

    clf = Perceptron()
    clf.fit(x_train, y_train)

    print("\n\t >> Predicting:")
    y_pred = clf.predict(x_test)

    print("\n\t >> Cross Cal Score:")
    scores = cross_val_score(clf, x_test, y_test, scoring='accuracy')
    print('\n\t >> Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100, np.std(scores) * 100))

    print_confusion_matrix(y_test, y_pred)
