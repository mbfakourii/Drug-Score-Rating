import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from helpers.compute_score import print_confusion_matrix

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


def mlpclassifier(x_train, x_test, y_test, y_train):
    print("\n\t >> Fit in Model:")

    model = MLPClassifier(solver='adam', alpha=0.01, activation='relu', hidden_layer_sizes=(150, 75), max_iter=1000,
                        random_state=1, batch_size=27017, learning_rate_init=0.001)
    model.fit(x_train, y_train)

    print("\n\t >> Predicting:")
    y_pred = model.predict(x_test)

    print("\n\t >> Cross Cal Score:")
    scores = cross_val_score(model, x_test, y_test, scoring='accuracy')
    print('\n\t >> Cross-validation mean accuracy {0:.2f}%, std {1:.2f}.'.format(np.mean(scores) * 100,
                                                                                 np.std(scores) * 100))

    print_confusion_matrix(y_test, y_pred)
