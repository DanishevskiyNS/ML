import numpy as np

from tree import CART


class RandomForest:
    def __init__(self, n_trees, max_samples, max_depth):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X):
        trees = [self._make_tree(self.max_depth) for _ in range(self.n_trees)]

        for tree in trees:
            X_s = self._get_random_subset(X, self.max_samples)
            tree.fit(X_s)
            self.trees.append(tree)

        return self

    def predict(self, X):
        y_preds = np.array([tree.predict(X) for tree in self.trees]).T
        y_voted = []

        for y in y_preds:
            values, counts = np.unique(y, return_counts=True)
            ind = np.argmax(counts)
            y_voted.append(values[ind])

        return y_voted
        
    @staticmethod
    def _get_random_subset(X, n_samples):
        random_indices = np.random.permutation(len(X))[:n_samples]

        return X[random_indices]

    @staticmethod
    def _make_tree(max_depth):
        tree = CART(max_depth)
        return tree
