import numpy as np


class DecisionTreeClassifier:

    def __init__(self, criterion='information_gain'):
        self.criterion = criterion
        self._eval_criterion = {}
        if criterion == 'information_gain':
            self._eval_criterion = self._information_gain
        elif criterion == 'gain_ratio':
            self._eval_criterion = self._gain_ratio
        elif criterion == 'gini_index':
            self._eval_criterion = self._gini_index

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.X = X
        self.y = y

        samples_idx = np.array(range(self.n_samples))
        features_idx = np.array(range(self.n_features))
        self._tree = self._step(samples_idx, features_idx)

    def _step(self, samples_idx, features_idx):
        node = {}

        # If all samples in 'samples_idx' have a same label 'y', 
        # return a leave node labeled with 'y'
        if self._all_same_class(samples_idx):
            node['prediction'] = self.y[samples_idx[0, 0]]
            node['is_leave'] = True
            return node

        # If the list of features is empty, return a leave node labeled with the most 
        # frequent class in 'samples_idx'
        if len(features_idx) == 0:
            node['prediction'] = self._most_frequent_class(samples_idx)
            node['is_leave'] = True
            return node

        # Select feature with best criterion division
        best_feature = self._choose_best_feature(samples_idx, features_idx)
        
        # Assign best feature to new node
        node['feature'] = best_feature

        # Remove best feature from the list of features 'features_idx'
        new_features_idx = list(features_idx)
        new_features_idx.remove(best_feature)
        new_features_idx = np.array(new_features_idx)

        node['children'] = []
        # Collect possible values of 'best_feature' feature, 
        # considering only the samples in 'samples_idx'
        A_values = np.unique(self.X[samples_idx, best_feature])
        # For each distinct value of feature 'best_feature'...
        for value in A_values:
            # Collect only samples with 'best_feature' value equal to 'value'
            new_samples_idx = self.X[samples_idx, best_feature] == value
            new_samples_idx = samples_idx[new_samples_idx]
            
            # If 'new_samples_idx' is empty, return a leave node labeled with the most
            # frequent class in 'new_samples_idx'
            if len(new_samples_idx) == 0:
                node['prediction'] = self._most_frequent_class(new_samples_idx)
                node['is_leave'] = True
                return node

            # Create recursively one subtree for each possible value
            # of 'best_feature' feature 
            node['is_leave'] = False
            child = {'value': value,
                     'feature': best_feature,
                     'is_leave': False, 
                     'tree': self._step(new_samples_idx, new_features_idx)}
            node['children'].append(child)

        return node

    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        i = 0
        node = self._tree
        while i < x.shape[0]:
            if node['is_leave']:
                y_pred[i]= node['prediction']
                i += 1
                node = self._tree
            else:
                for child in node['children']:
                    feature = node['feature']
                    value = child['value']
                    if x[i, feature] == value:
                        if child['is_leave']:
                            y_pred[i] = child['prediction']
                            i += 1
                            node = self._tree
                        else:
                            node = child['tree']
                        break
        return y_pred

    def _all_same_class(self, samples_idx):
        for classe in self.classes:
            count = sum(self.y[samples_idx] == classe)
            if count == self.n_samples:
                return True
        return False

    def _most_frequent_class(self, samples_idx):
        clas = None
        max_samples = -np.inf
        for classe in self.classes:
            count = sum(self.y[samples_idx] == classe)
            if count > max_samples:
                max_samples = count
                clas = classe
        return clas

    def _choose_best_feature(self, samples_idx, features_idx):
        best_feature = None
        max_gain = -np.inf
        for feature in features_idx:
            gain = self._eval_criterion(samples_idx, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature

    def _entropy(self, samples_idx, feature=None):
        info = 0
        n_total = len(samples_idx)
        
        if feature is None:
            for classe in self.classes:
                n_classe = sum(self.y[samples_idx] == classe)
                prob_classe = n_classe / n_total
                if prob_classe > 0:
                    info += prob_classe * np.log2(prob_classe)
            info = -info
        else:
            values = np.unique(self.X[samples_idx, feature])
            for value in values:
                n_value = sum(self.X[samples_idx, feature] == value)
                prob_value = n_value / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[np.where(self.X[samples_idx, feature] == value)]
                    info += prob_value * self._entropy(feature_samples_idx)
        
        return info

    def _split_info(self, samples_idx, feature):
        info = 0
        n_total = len(samples_idx)
        
        values = np.unique(self.X[samples_idx, feature])
        for value in values:
            n_value = sum(self.X[samples_idx, feature] == value)
            prob_value = n_value / n_total
            if prob_value > 0:
                info += prob_value * np.log2(prob_value)
        
        return -info

    def _information_gain(self, samples_idx, feature):
        info_all = self._entropy(samples_idx)
        info_feature = self._entropy(samples_idx, feature)
        return info_all - info_feature

    def _gain_ratio(self, samples_idx, feature):
        split_info = self._split_info(samples_idx, feature)
        if split_info > 0:
            information_gain = self._information_gain(samples_idx, feature)
            return information_gain / split_info
        return 0

    def _gini(self, samples_idx):
        gini = 1
        n_total = len(samples_idx)

        for classe in self.classes:
            n_samples = sum(self.y[samples_idx])
            prob_classe = n_samples / n_total
            gini -= prob_classe ** 2

        return gini

    def _gini_index(self, samples_idx, feature):
        gini_all = self._gini(samples_idx)

        ginis_feature = {}
        n_total = len(samples_idx)
        values = np.unique(self.X[samples_idx, feature])
        for value in values:
            ginis_feature[value] = 0

            n_values = np.sum(self.X[samples_idx, feature] == value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[np.where(self.X[samples_idx, feature] == value)]
                ginis_feature[value] += prob_value * self._gini(feature_samples_idx)

            n_values = np.sum(self.X[samples_idx, feature] != value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[np.where(self.X[samples_idx, feature] != value)]
                ginis_feature[value] += prob_value * self._gini(feature_samples_idx)
        
        return gini_all - min(ginis_feature.values())
