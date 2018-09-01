import numpy as np
from itertools import count
from graphviz import Digraph


class DecisionTreeClassifier:

    def __init__(self, criterion='information_gain', max_depth=1000, 
                 one_split_by_feature=False, random_tree=False):
        self.max_depth = max_depth
        self.one_split_by_feature = one_split_by_feature
        self.random_tree = random_tree
        self.generate_id = count()
        self._evals = {'information_gain': self._information_gain, 
                       'gain_ratio': self._gain_ratio, 
                       'gini_index': self._gini_index}
        self._eval = self._evals[criterion]

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.X, self.y = X, y

        samples_idx = np.array(range(self.n_samples))
        features_idx = np.array(range(self.n_features))

        # If random tree then evaluate only sqrt('features_idx') features
        if self.random_tree:
            self.n_random_features = int(np.sqrt(features_idx.shape[0]))

        self._tree = self._step(samples_idx, features_idx)

    def _step(self, samples_idx, features_idx, depth=0):
        node = {}
        node['n_samples'] = samples_idx.shape[0]
        node['is_leaf'] = False
        node['id'] = next(self.generate_id)

        # If all samples in 'samples_idx' have a same label 'y', 
        # return a leaf node labeled with 'y'
        if self._all_same_class(samples_idx):
            node['prediction'] = self.y[samples_idx[0]]
            node['is_leaf'] = True
            return node

        # If the list of features is empty or depth equals to max depth
        # return a leaf node labeled with the most 
        # frequent class in 'samples_idx'
        if len(features_idx) == 0 or depth == self.max_depth:
            node['prediction'] = self._most_frequent_class(samples_idx)
            node['is_leaf'] = True
            return node           

        # Select feature with best criterion division
        best_feature, value = self._choose_best_feature(samples_idx, 
                                                        features_idx)

        if self.random_tree:
            # Do not change available features (always will be the whole set)
            new_features_idx = features_idx
        else:
            new_features_idx = list(features_idx)
            # Remove best feature from the list of features 'features_idx'
            if self.one_split_by_feature:
                new_features_idx.remove(best_feature)
            new_features_idx = np.array(new_features_idx)

        # Collect cut points of 'best_feature' feature, 
        # considering only the samples in 'samples_idx'
        kind = self._get_feature_kind(best_feature)
        cuts = self._select_cut_points(samples_idx, best_feature, kind)

        node['out_feature'] = best_feature
        node['kind'] = kind
        node['eval'] = round(value, 3)

        if kind == 'categorical':
            node['children'] = []

            # For each distinct category value...
            for value in cuts:
                # Collect only samples with 'best_feature' value 
                # equal to 'value'
                new_samples_idx = self.X[samples_idx, best_feature] == value
                new_samples_idx = samples_idx[new_samples_idx]

                # Create recursively one subtree for each distinct category 
                # value of 'best_feature' feature 
                child = self._step(new_samples_idx, new_features_idx, depth + 1)
                child['in_value'] = value
                node['children'].append(child)

        elif kind == 'numeric':
            # There is only one cut point
            cut_point = cuts[0]

            # Collect only samples with 'best_feature' value 
            # less or equal than 'cut_point'
            idx_left = self.X[samples_idx, best_feature] <= cut_point
            new_samples_idx_left = samples_idx[idx_left]

            # Create recursively one subtree for samples with 
            # 'best_feature' value <= cut point 
            child = self._step(new_samples_idx_left, new_features_idx, 
                               depth + 1)
            child['in_value'] = cut_point
            node['child_left'] = child

            # Collect only samples with 'best_feature' value 
            # greater than 'cut_point'
            idx_right = self.X[samples_idx, best_feature] > cut_point
            new_samples_idx_right = samples_idx[idx_right]

            # Create recursively one subtree for samples with 
            # 'best_feature' value > cut point
            child = self._step(new_samples_idx_right, new_features_idx, 
                               depth + 1)
            child['in_value'] = cut_point 
            node['child_right'] = child
        
        return node

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            x = X[i, :]
            y_pred[i] = self._predict(x, self._tree)
        return y_pred

    def _predict(self, x, node):
        if node['is_leaf']:
            return node['prediction']
        
        feature = node['out_feature']
        kind = node['kind']
        
        if kind == 'categorical':
            for child in node['children']:
                value = child['in_value']
                if x[feature] == value:
                    return self._predict(x, child)
            # if 'x[feature]' value is not in tree, 
            # then choose the child with more samples
            max_child = None
            max_samples = -np.inf
            for child in node['children']:
                if child['n_samples'] > max_samples:
                    max_child = child
                    max_samples = child['n_samples']
            return self._predict(x, max_child)
        elif kind == 'numeric':
            value = node['child_left']['in_value']
            if x[feature] <= value:
                return self._predict(x, node['child_left'])
            else:
                return self._predict(x, node['child_right'])

    def _all_same_class(self, samples_idx):
        for classe in self.classes:
            count = sum(self.y[samples_idx] == classe)
            if count == samples_idx.shape[0]:
                return True
        return False

    def _most_frequent_class(self, samples_idx):
        max_classe = None
        max_samples = -np.inf
        for classe in self.classes:
            count = sum(self.y[samples_idx] == classe)
            if count > max_samples:
                max_samples = count
                max_classe = classe
        return max_classe

    def _choose_best_feature(self, samples_idx, features_idx):
        # If random tree then evaluate only sqrt('features_idx') features
        if self.random_tree:
            idx = np.arange(0, features_idx.shape[0])
            np.random.shuffle(idx)
            features_idx = features_idx[idx[:self.n_random_features]]

        best_feature = None
        max_gain = -np.inf
        for feature in features_idx:
            gain = self._eval(samples_idx, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature, max_gain

    def _get_feature_kind(self, feature):
        kind = ''
        if self.X[:, feature].dtype == 'int64':
            kind = 'categorical'
        elif self.X[:, feature].dtype == 'float64':
            kind = 'numeric'
        return kind

    def _select_cut_points(self, samples_idx, feature, kind):
        cuts = []
        if kind == 'categorical':
            # Select all distinct values of this features as cuts
            cuts = np.unique(self.X[samples_idx, feature])
        elif kind == 'numeric':
            # Select mean point as cut point of this feature
            cuts.append(round(np.mean(self.X[samples_idx, feature]), 3))
        return cuts

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
            kind = self._get_feature_kind(feature)
            values = self._select_cut_points(samples_idx, feature, kind)

            if kind == 'categorical':
                for value in values:
                    n_value = sum(self.X[samples_idx, feature] == value)
                    idx = np.where(self.X[samples_idx, feature] == value)
                    prob_value = n_value / n_total
                    if prob_value > 0:
                        feature_samples_idx = samples_idx[idx]
                        info += prob_value * self._entropy(feature_samples_idx)
            elif kind == 'numeric':
                value = values[0]

                n_value = sum(self.X[samples_idx, feature] <= value)
                idx = np.where(self.X[samples_idx, feature] <= value)
                prob_value = n_value / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[idx]
                    info += prob_value * self._entropy(feature_samples_idx)

                n_value = sum(self.X[samples_idx, feature] > value)
                idx = np.where(self.X[samples_idx, feature] > value)
                prob_value = n_value / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[idx]
                    info += prob_value * self._entropy(feature_samples_idx)
        
        return info

    def _split_info(self, samples_idx, feature):
        info = 0
        n_total = len(samples_idx)
        
        kind = self._get_feature_kind(feature)
        values = self._select_cut_points(samples_idx, feature, kind)

        if kind == 'categorical':
            for value in values:
                n_value = sum(self.X[samples_idx, feature] == value)
                prob_value = n_value / n_total
                if prob_value > 0:
                    info += prob_value * np.log2(prob_value)
        elif kind == 'numeric':
            value = values[0]

            n_value = sum(self.X[samples_idx, feature] <= value)
            prob_value = n_value / n_total
            if prob_value > 0:
                info += prob_value * np.log2(prob_value)
            
            n_value = sum(self.X[samples_idx, feature] > value)
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

        for _ in self.classes:
            n_samples = sum(self.y[samples_idx])
            prob_classe = n_samples / n_total
            gini -= prob_classe ** 2

        return gini

    def _gini_index(self, samples_idx, feature):
        gini_all = self._gini(samples_idx)

        ginis_feature = {}
        n_total = len(samples_idx)
        kind = self._get_feature_kind(feature)
        values = self._select_cut_points(samples_idx, feature, kind)
        
        if kind == 'categorical':
            for value in values:
                ginis_feature[value] = 0

                n_values = np.sum(self.X[samples_idx, feature] == value)
                idx = np.where(self.X[samples_idx, feature] == value)
                prob_value = n_values / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[idx]
                    aux = prob_value * self._gini(feature_samples_idx)
                    ginis_feature[value] += aux

                n_values = np.sum(self.X[samples_idx, feature] != value)
                idx = np.where(self.X[samples_idx, feature] != value)
                prob_value = n_values / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[idx]
                    aux = prob_value * self._gini(feature_samples_idx)
                    ginis_feature[value] += aux
        elif kind == 'numeric':
            value = values[0]
            ginis_feature[value] = 0

            n_values = np.sum(self.X[samples_idx, feature] <= value)
            idx = np.where(self.X[samples_idx, feature] <= value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[idx]
                aux = prob_value * self._gini(feature_samples_idx)
                ginis_feature[value] += aux

            n_values = np.sum(self.X[samples_idx, feature] > value)
            idx = np.where(self.X[samples_idx, feature] > value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[idx]
                aux = prob_value * self._gini(feature_samples_idx)
                ginis_feature[value] += aux
        
        return gini_all - min(ginis_feature.values())

    def show_decision_tree(self):
        colors = {0: '#ff000055', 1: '#00ff0055', 2: '#0000ff55', 
                  3: '#ffff0055', 4: '#ff00ff55', 5: '#00ffff55'}

        nodes, edges = [], []
        self._construct_tree(self._tree, nodes, edges)

        decision_tree = Digraph(filename='tree')
        decision_tree.edge_attr.update(arrowhead='vee')

        for node in nodes:
            label = ''

            id = str(node['id'])
            n_samples = str(node['n_samples'])

            label += f'node_{id}\n'
            label += f'n_samples = {n_samples}\n'

            if node['is_leaf']:
                prediction = node['prediction']
                label += f"predicted class = {prediction}"
                color = colors[prediction]
                shape = 'circle'
            else:
                eval = str(node['eval'])
                label += f'eval = {eval}\n'
                shape = 'square'
                color = '#77777755'

            decision_tree.node(name=id, label=label, shape=shape, 
                               fillcolor=color, style='filled')

        for edge in edges:
            tail_name = str(edge[0])
            head_name = str(edge[1])
            label = edge[2]
            decision_tree.edge(tail_name=tail_name, head_name=head_name, 
                               label=label)
        
        decision_tree.render()

    def _construct_tree(self, root, nodes, edges):
        nodes.append(root)

        if root['is_leaf']:
            return

        kind = root['kind']
        if kind == 'categorical':
            for child in root['children']:
                id_root = root['id']
                id_child = child['id']
                out_feature = root['out_feature']
                in_value = child['in_value']
                label = f'feature_{out_feature} == {in_value}'
                edges.append((id_root, id_child, label))
                self._construct_tree(child, nodes, edges)
        elif kind == 'numeric':
            id_root = root['id']
            child_left = root['child_left']
            child_right = root['child_right']
            id_child_left = child_left['id']
            id_child_right = child_right['id']
            out_feature = root['out_feature']
            in_value_left = child_left['in_value']
            in_value_right = child_right['in_value']
            label_left = f'feature_{out_feature} <= {in_value_left}'
            label_right = f'feature_{out_feature} > {in_value_right}'
            edges.append((id_root, id_child_left, label_left))
            edges.append((id_root, id_child_right, label_right))
            self._construct_tree(child_left, nodes, edges)
            self._construct_tree(child_right, nodes, edges)
