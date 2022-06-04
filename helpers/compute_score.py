from sklearn.metrics import confusion_matrix
import numpy as np


def print_confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print('\n\t >>> True positive = ', tp)
    print('\n\t >>> False positive = ', fp)
    print('\n\t >>> False negative = ', fn)
    print('\n\t >>> True negative = ', tn)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print('\n\t >>> F-measure Accuracy = {0:.2f}% | Precision = {1:.2f} |  Recall = {2:.2f}'.format(
        np.mean(accuracy) * 100, np.mean(precision) * 100, np.mean(recall) * 100))
