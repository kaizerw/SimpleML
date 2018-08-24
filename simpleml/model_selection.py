import numpy as np
from random import choices


def train_test_split(X, y, train_size=0.7, shuffle=True):
    if shuffle:
        idx = list(range(y.shape[0]))
        np.random.shuffle(idx)
        X = X[idx, :]
        y = y[idx]
    classes = np.unique(y)
    train_idx, test_idx = [], []
    for classe in classes:
        idx_samples = np.where(y == classe)[0]
        n_train = int(train_size * len(idx_samples))
        train_idx.extend(idx_samples[:n_train])
        test_idx.extend(idx_samples[n_train:])    
    return X[train_idx, :], X[test_idx, :], y[train_idx], y[test_idx]


def stratified_k_fold(model, metric, X, y, n_folds=5, repetitions=5, shuffle=True):
    classes = np.unique(y)

    evaluations = []
    for r in range(repetitions):
        if shuffle:
            idx = list(range(y.shape[0]))
            np.random.shuffle(idx)
            X = X[idx, :]
            y = y[idx]

        idx_samples_per_class = {}
        for classe in classes:
            idx_samples_per_class[classe] = np.where(y == classe)[0]

        for fold in range(n_folds):
            idx_train = []
            idx_test = []
            for classe in classes:
                n_samples = len(idx_samples_per_class[classe]) / n_folds
                begin = int(fold * n_samples)
                last = int(begin + n_samples)
                idx_test.extend(idx_samples_per_class[classe][begin:last])
                idx_train.extend(list(set(idx_samples_per_class[classe]) - set(idx_test)))

            X_train = X[idx_train, :]
            y_train = y[idx_train]
            X_test = X[idx_test, :]
            y_test = y[idx_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true = y_test
            evaluation = metric(y_true, y_pred)

            evaluations.append(evaluation)

    return evaluations


def leave_one_out(model, metric, X, y, shuffle=True):
    if shuffle:
        idx = list(range(y.shape[0]))
        np.random.shuffle(idx)
        X = X[idx, :]
        y = y[idx]

    classes = np.unique(y)
    n_samples = y.shape[0]

    evaluations = []
    for sample in range(n_samples):
        idx_train = list(range(n_samples))
        idx_train.remove(sample)
        idx_test = [sample]
        
        X_train = X[idx_train, :]
        y_train = y[idx_train]
        X_test = X[idx_test, :]
        y_test = y[idx_test]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true = y_test
        evaluation = metric(y_true, y_pred)

        evaluations.append(evaluation)

    return evaluations


def bootstrap(model, metric, X, y, n_folds=5, repetitions=5, train_size=0.7):
    size_train = int(y.shape[0] * train_size)
    idx = list(range(y.shape[0]))

    evaluations = []

    for r in range(repetitions):
        for fold in range(n_folds):
            np.random.shuffle(idx)
            X = X[idx, :]
            y = y[idx]

            idx_train = choices(idx, k=size_train)
            idx_test = (set(idx) - set(idx_train))

            X_train = X[idx_train, :]
            y_train = y[idx_train]
            X_test = X[idx_test, :]
            y_test = y[idx_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true = y_test
            evaluation = metric(y_true, y_pred)

            evaluations.append(evaluation)

    return evaluations
