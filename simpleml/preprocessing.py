import numpy as np


class MinMaxScaler:

    def __init__(self):
        pass

    def fit(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self
    
    def transform(self, X):
        return (X - self.min) / (self.max - self.min)


class StandardScaler:
    
    def __init__(self):
        pass

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mu) / self.sigma
