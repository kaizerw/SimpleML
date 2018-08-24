import numpy as np
from copy import deepcopy


class OneVsRestClassifier:

    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.classifiers = [deepcopy(self.model) 
                            for _ in range(self.n_classes)]
        self.X = X
        self.y = y

        for classe in self.classes:
            y_true = np.zeros(self.n_samples)
            y_true[y == classe] = 1
            self.classifiers[classe].fit(X, y_true)

    def predict(self, X):
        n_samples_pred = X.shape[0]
        y_pred = np.zeros(n_samples_pred)
        for i in range(n_samples_pred):
            x = np.reshape(X[i, :], (1, -1))
            y_pred[i] = np.argmax([classifier._activation(x)
                                   for classifier in self.classifiers])
        return y_pred
