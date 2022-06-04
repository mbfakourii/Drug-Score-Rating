import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from helpers.compute_score import print_confusion_matrix


def logistic_regression(x_train, x_test, y_test, y_train):
    print("\n\t >> Fit in Model:")

    model = LogisticRegression(random_state=0,max_iter=1000)
    model.fit(x_train, y_train)

    print("\n\t >> Predicting:")
    y_pred = model.predict(x_test)

    print("\n\t >> Cross Cal Score:")
    scores = cross_val_score(model, x_test, y_test, scoring='accuracy')
    print('\n\t >> Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100,
                                                                                 np.std(scores) * 100))

    print_confusion_matrix(y_test, y_pred)
