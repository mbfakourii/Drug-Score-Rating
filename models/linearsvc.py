from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import numpy as np
from helpers.compute_score import print_confusion_matrix


def linearsvc(x_train, x_test, y_test, y_train):
    print("\n\t >> Fit in Model:")

    model = LinearSVC(C=1, class_weight="balanced")
    model.fit(x_train, y_train)

    print("\n\t >> Predicting:")
    y_pred = model.predict(x_test)

    print("\n\t >> Cross Cal Score:")
    scores = cross_val_score(model, x_test, y_test, scoring='accuracy')
    print('\n\t >> Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100,
                                                                                 np.std(scores) * 100))
    print_confusion_matrix(y_test, y_pred)
