import numpy as np
from scipy.stats import mode
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KNNClassifier:

    def __init__(self, k=5, p=2):
        self.k = k # Number of neighbors to considerate in voting
        self.p = p # Power parameter to Minkowski distance

    # Minkowski distance
    def metric(self, x1, x2):
        return sum(abs(i - j) ** self.p for i, j in zip(x1, x2)) ** (1 / self.p)

    def predict(self, X_train, y_train, X_test):
        y_pred = np.zeros(X_test.shape[0])
        
        for i in range(X_test.shape[0]):
            # Calculate dists from all samples to X_test[i]
            dists = []
            for j in range(X_train.shape[0]):
                dists.append([j, self.metric(X_train[j, :], X_test[i, :])])
            
            # Collecting the k closest samples
            dists.sort(key=lambda i: i[1])
            preds = [y_train[i[0]] for i in dists[:self.k]]
            
            # Predict the class with more samples
            y_pred[i] = mode(preds)[0]

        return y_pred


if __name__ == '__main__':
    X, y = make_blobs(n_samples=500, n_features=10, centers=5) 

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    for k in [1, 3, 5, 7, 9]:
        model = KNNClassifier(k=k)
        print(f'Accuracy with k={k}:', sum(model.predict(X, y, X) == y.T) / y.shape[0])
