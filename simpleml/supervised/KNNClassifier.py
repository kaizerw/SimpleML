import numpy as np


class KNNClassifier:

    def __init__(self, k=5, p=2):
        self.k = k # Number of neighbors to considerate in voting
        self.p = p # Power parameter to Minkowski distance

    # Minkowski distance
    def _metric(self, x1, x2):
        return sum(abs(i - j) ** self.p for i, j in zip(x1, x2)) ** (1 / self.p)

    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.X, self.y = X, y

    def predict(self, X):
        y_pred = []
        
        for i in range(X.shape[0]):
            # Calculate distances from all samples to X[i]
            distances = []
            for j in range(self.n_samples):
                distances.append([j, self._metric(self.X[j, :], X[i, :])])
            
            # Collect the k closest samples
            distances.sort(key=lambda i: i[1])
            predictions = [self.y[i[0]] for i in distances[:self.k]]
            
            # Predict the class with more samples
            y_pred.append(np.argmax(np.bincount(predictions)))

        return np.array(y_pred)
