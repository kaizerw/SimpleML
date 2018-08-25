import numpy as np


class KNNRegressor:

    def __init__(self, k=5, p=2):
        self.k = k # Number of neighbors to considerate in mean
        self.p = p # Power parameter to Minkowski distance

    # Minkowski distance
    def _metric(self, x1, x2):
        return sum(abs(i - j) ** self.p for i, j in zip(x1, x2)) ** (1 / self.p)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        
        for i in range(X_test.shape[0]):
            # Calculate dists from all samples to X_test[i]
            dists = []
            for j in range(self.X_train.shape[0]):
                dists.append([j, self._metric(self.X_train[j, :], X_test[i, :])])
            
            # Collect the k closest samples
            dists.sort(key=lambda i: i[1])
            preds = [self.y_train[i[0]] for i in dists[:self.k]]
            
            # Predict target with the mean of the neighbors' targets
            y_pred[i] = np.mean(preds)

        return y_pred
