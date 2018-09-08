import numpy as np


def confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes))
    for t in classes:
        for p in classes:
            matrix[t, p] = sum((y_true == t) & (y_pred == p))
    return matrix


def accuracy(y_true, y_pred):
    n = y_true.shape[0]
    tp = 0
    for classe in np.unique(y_true):
        tp += sum((y_true == classe) & (y_pred == classe))
    return (tp) / (n + 1e-10) 


def recall(y_true, y_pred, kind='macro'):
    classes = np.unique(y_true)

    if kind == 'micro':
        tp = 0
        fn = 0
        for classe in classes:
            tp += sum((y_true == classe) & (y_pred == classe))
            fn += sum((y_true == classe) & (y_pred != classe))
        return tp / (tp + fn + 1e-10)

    recalls = []
    for classe in classes:
        tp = sum((y_true == classe) & (y_pred == classe))
        fn = sum((y_true == classe) & (y_pred != classe))
        recalls.append(tp / (tp + fn + 1e-10))
    
    if kind == 'macro':
        return np.mean(recalls)
    elif kind == 'all':
        return np.array(recalls)


def precision(y_true, y_pred, kind='macro'):
    classes = np.unique(y_true)

    if kind == 'micro':
        tp = 0
        fp = 0
        for classe in classes:
            tp = sum((y_true == classe) & (y_pred == classe))
            fp = sum((y_true != classe) & (y_pred == classe))
        return tp / (tp + fp + 1e-10)

    precisions = []
    for classe in classes:
        tp = sum((y_true == classe) & (y_pred == classe))
        fp = sum((y_true != classe) & (y_pred == classe))
        precisions.append(tp / (tp + fp + 1e-10))
    
    if kind == 'macro':
        return np.mean(precisions)
    elif kind == 'all':
        return np.array(precisions)


def f1_score(y_true, y_pred, kind='macro'):
    classes = np.unique(y_true)

    f1_scores = []
    for _ in classes:
        prec = precision(y_true, y_pred, kind=kind)
        rec = recall(y_true, y_pred, kind=kind)
        f1_scores.append(2 * (prec * rec) / (prec + rec + 1e-10))
    
    return np.mean(f1_scores, axis=0)


def mean_absolute_error(y_true, y_pred):
    n = y_true.shape[0]
    return sum(abs(y_true - y_pred)) / (n + 1e-10)


def mean_squared_error(y_true, y_pred):
    n = y_true.shape[0]
    return sum((y_true - y_pred) ** 2) / (n + 1e-10)


def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)

    ss_tot = sum((y_true - y_mean) ** 2) + 1e-10
    ss_res = sum((y_true - y_pred) ** 2) + 1e-10

    return 1 - (ss_res / ss_tot)


def log_loss_score(y_true, y_pred):
    n = y_true.shape[0]
    return -(1 / n) * (np.sum(y_true * np.log(y_pred) + 
                              (1 - y_true) * np.log(1 - y_pred)))


def zero_one_loss(y_true, y_pred):
    return sum(y_true != y_pred)
