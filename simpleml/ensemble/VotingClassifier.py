import numpy as np


class VotingClassifier:

    def __init__(self, models):
        self.models = models # list of models to poll

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []

        for i in range(n_samples):
            x = np.reshape(X[i, :], (1, -1))
            predictions = [model.predict(x)[0] for model in self.models]
            y_pred.append(np.argmax(np.bincount(predictions)))
        return np.array(y_pred)
