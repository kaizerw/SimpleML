import numpy as np

from ..supervised.DecisionTreeClassifier import DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(self, n_trees=10):
        self.trees = [DecisionTreeClassifier(random_tree=True) 
                       for _ in range(n_trees)]
        self.n_trees = n_trees
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for model in self.trees:
            bag_idx = np.random.randint(0, n_samples, n_samples)
            model.fit(X[bag_idx, :], y[bag_idx])

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            x = np.reshape(X[i, :], (1, -1))
            predictions = [model.predict(x)[0] for model in self.trees]
            y_pred.append(np.argmax(np.bincount(predictions)))
        return np.array(y_pred)
