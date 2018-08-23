import numpy as np
from sklearn.datasets import make_classification


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


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print('Train:', np.bincount(y_train))
    print('Test:', np.bincount(y_test))
