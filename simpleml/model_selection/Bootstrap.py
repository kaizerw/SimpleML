import numpy as np
from sklearn.datasets import make_classification


def accuracy(y_true, y_pred):
    n = y_true.shape[0]
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0)) 
    return (tp + tn) / n 


def bootstrap(model, metric, X, y, n_folds=5, repetitions=5, train_size=0.7):
    size_train = int(y.shape[0] * train_size)
    idx = list(range(y.shape[0]))

    evaluations = []

    for r in range(repetitions):
        for fold in range(n_folds):
            np.random.shuffle(idx)
            X = X[idx, :]
            y = y[idx]

            idx_train = idx[:size_train]
            idx_test = idx[size_train:]

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


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    metric = accuracy

    result = bootstrap(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')
