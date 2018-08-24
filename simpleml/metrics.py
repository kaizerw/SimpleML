import numpy as np


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((2, 2))
    for t in range(2):
        for p in range(2):
            matrix[t, p] = sum((y_true == t) & (y_pred == p))
    return matrix


def accuracy(y_true, y_pred):
    n = y_true.shape[0]
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0)) 
    return (tp + tn) / n 


def error(y_true, y_pred):
    n = y_true.shape[0]
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return (fp + fn) / n


def recall(y_true, y_pred):
    n = y_true.shape[0]
    tp = sum((y_true == 1) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)


def precision(y_true, y_pred):
    n = y_true.shape[0]
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)


def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec)


def mean_absolute_error(y_true, y_pred):
    n = y_true.shape[0]
    return sum(abs(y_true - y_pred)) / n


def mean_squared_error(y_true, y_pred):
    n = y_true.shape[0]
    return sum((y_true - y_pred) ** 2) / n


def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)

    ss_tot = sum((y_true - y_mean) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)

    return 1 - (ss_res / ss_tot)
