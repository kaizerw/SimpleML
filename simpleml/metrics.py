import numpy as np
from sklearn.datasets import make_classification, make_regression
import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    from sklearn import metrics

    from supervised import LogisticRegression
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LogisticRegression()
    model.fit(X, y)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = model.predict(X)
    y_true = y

    print('Confusion Matrix:', confusion_matrix(y_true, y_pred) == metrics.confusion_matrix(y_true, y_pred), sep='\n')
    print('Accuracy:', accuracy(y_true, y_pred) - metrics.accuracy_score(y_true, y_pred))
    print('Error:', error(y_true, y_pred) - (1 - metrics.accuracy_score(y_true, y_pred)))
    print('Recall:', recall(y_true, y_true) - metrics.recall_score(y_true, y_pred))
    print('Precision:', precision(y_true, y_pred) - metrics.precision_score(y_true, y_pred))
    print('F1-score:', f1_score(y_true, y_pred) - metrics.f1_score(y_true, y_pred))

    print('*' * 80)

    from supervised import LinearRegression
    X, y = make_regression(n_samples=500, n_features=10, 
                           n_informative=10, n_targets=1) 
    model = LinearRegression()
    model.fit(X, y)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = model.predict(X)
    y_true = y

    print('mean_absolute_error:', mean_absolute_error(y_true, y_pred) - metrics.mean_absolute_error(y_true, y_pred))
    print('mean_squared_error:', mean_squared_error(y_true, y_pred) - metrics.mean_squared_error(y_true, y_pred))
    print('r2_score:', r2_score(y_true, y_pred) - metrics.r2_score(y_true, y_pred))
