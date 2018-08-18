import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class DecisionTreeClassifier:

    def __init__(self, criterion='information_gain', max_depth=None):
        self.criterion = criterion
        self._eval_criterion = {}
        self._eval_criterion['information_gain'] = self._information_gain
        self._eval_criterion['gain_ratio'] = self._gain_ratio
        self._eval_criterion['gini_index'] = self._gini_index
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        y = y.reshape((-1, 1))
        self.classes = np.unique(y)
        self.X = X
        self.y = y

        samples_idx = np.array(range(self.n_samples))
        features_idx = np.array(range(self.n_features))
        self._tree = self._step(samples_idx, features_idx)

    def _step(self, samples_idx, features_idx):
        node = {}

        if self._all_same_class(samples_idx):
            node['prediction'] = self.y[samples_idx[0, 0]]
            node['is_leave'] = True
            return node

        if len(features_idx) == 0:
            node['prediction'] = self._most_frequent_class(samples_idx)
            node['is_leave'] = True
            return node

        best_feature = self._choose_best_feature(samples_idx, features_idx)
        
        node['feature'] = best_feature
        new_features_idx = list(features_idx)
        new_features_idx.remove(best_feature)
        new_features_idx = np.array(new_features_idx)

        node['children'] = []

        A_values = np.unique(self.X[samples_idx, best_feature])
        for value in A_values:
            new_samples_idx = np.where(self.X[samples_idx, best_feature]==value)[0]
            new_samples_idx = samples_idx[new_samples_idx]
            
            if len(new_samples_idx) == 0:
                node['prediction'] = self._most_frequent_class(new_samples_idx)
                node['is_leave'] = True
                return node

            node['is_leave'] = False
            child = {'value': value,
                     'feature': best_feature,
                     'is_leave': False, 
                     'tree': self._step(new_samples_idx, new_features_idx)}
            node['children'].append(child)

        return node

    def predict(self, x):
        y_pred = []
        i = 0
        node = self._tree
        while i < x.shape[0]:
            if node['is_leave']:
                y_pred.append(node['prediction'])
                i += 1
                node = self._tree
            else:
                for child in node['children']:
                    feature = node['feature']
                    value = child['value']
                    if x[i, feature] == value:
                        if child['is_leave']:
                            y_pred.append(child['prediction'])
                            i += 1
                            node = self._tree
                        else:
                            node = child['tree']
                        break
        return np.reshape(np.array(y_pred), (-1, 1))

    def _all_same_class(self, samples_idx):
        for classe in self.classes:
            count = sum(self.y[samples_idx, 0]==classe)
            if count == self.n_samples:
                return True
        return False

    def _most_frequent_class(self, samples_idx):
        clas = -1
        max_samples = -1
        for classe in self.classes:
            count = sum(self.y[samples_idx, 0]==classe)
            if count > max_samples:
                max_samples = count
                clas = classe
        return clas

    def _choose_best_feature(self, samples_idx, features_idx):
        best_feature = -1
        max_gain = -1
        for feature in features_idx:
            gain = self._eval_criterion[self.criterion](samples_idx, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature

    def _information_gain(self, samples_idx, feature):
        return 0
        
    def _gain_ratio(self, samples_idx, feature):
        return 0

    def _gini_index(self, samples_idx, feature):
        return 0

    
if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)
    y = y.reshape((-1, 1))

    # By now, only works with categorical features
    # TODO: Generalize to numeric and mixed features
    X = abs(X.astype(int))

    model = DecisionTreeClassifier()
    model.fit(X, y)

    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred==y) / y.shape[0]))
