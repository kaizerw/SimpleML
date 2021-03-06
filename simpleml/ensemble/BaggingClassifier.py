import numpy as np
from copy import deepcopy


class BaggingClassifier:

    def __init__(self, base_model, n_models=10):
        self.models = [deepcopy(base_model) for _ in range(n_models)]
        self.n_models = n_models
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for model in self.models:
            bag_idx = np.random.randint(0, n_samples, n_samples)
            model.fit(X[bag_idx, :], y[bag_idx])

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            x = np.reshape(X[i, :], (1, -1))
            predictions = [model.predict(x)[0] for model in self.models]
            y_pred.append(np.argmax(np.bincount(predictions)))
        return np.array(y_pred)
