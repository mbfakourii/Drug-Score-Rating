import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB

from helpers.compute_score import print_confusion_matrix


def bernoullinb(x_train, x_test, y_test, y_train):
    max = 0
    bestAlpha = 0.1
    for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        clf = BernoulliNB(alpha=a)
        scores = cross_val_score(estimator=clf, X=x_train, y=y_train)
        score = sum(scores) / len(scores)
        if score > max:
            max = score
            bestAlpha = a

    print("\n\t >> Fit in Model:")

    clf = BernoulliNB(alpha=bestAlpha)
    clf.fit(x_train, y_train)

    print("\n\t >> Predicting:")
    y_pred = clf.predict(x_test)

    print("\n\t >> Cross Cal Score:")
    scores = cross_val_score(clf, x_test, y_test, scoring='accuracy')
    print('\n\t >> Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100,
                                                                                 np.std(scores) * 100))

    print_confusion_matrix(y_test, y_pred)
