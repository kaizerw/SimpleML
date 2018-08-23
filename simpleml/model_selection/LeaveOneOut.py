import numpy as np
from sklearn.datasets import make_classification


def accuracy(y_true, y_pred):
    n = y_true.shape[0]
    tp = sum((y_true == 1) & (y_pred == 1))
    tn = sum((y_true == 0) & (y_pred == 0)) 
    return (tp + tn) / n 


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


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    metric = accuracy

    result = leave_one_out(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')
