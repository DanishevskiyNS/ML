import numpy as np


def unique_vals(X, col):
    return set([row[col] for row in X])

def class_counts(X):
    counts = {}

    for row in X:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    return counts

def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="

        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class CART:
    def __init__(self, max_depth):
        self.tree = None
        self.max_depth = max_depth
    
    def partition(self, X, question):
        true_X, false_X = [], []
        
        for row in X:
            depth = max(len(true_X), len(false_X))
            if depth < self.max_depth:
                if question.match(row):
                    true_X.append(row)
                else:
                    false_X.append(row)

        return true_X, false_X

    def gini(self, X):
        counts = class_counts(X)
        impurity = 1

        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(X))
            impurity -= prob_of_lbl**2

        return impurity

    def info_gain(self, left, right, current_uncertainty):
        p = float(len(left)) / (len(left) + len(right))

        return current_uncertainty - p * self.gini(left) - (1 - p) * self.gini(right)

    def find_best_split(self, X):
        best_gain = 0
        best_question = None
        current_uncertainty = self.gini(X)
        n_features = len(X[0]) - 1

        for col in range(n_features):
            values = set([row[col] for row in X])

            for val in values:
                question = Question(col, val)
                true_X, false_X = self.partition(X, question)
                if len(true_X) == 0 or len(false_X) == 0:
                    continue

                gain = self.info_gain(true_X, false_X, current_uncertainty)
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question
    
    def classify(self, X, node):
        if isinstance(node, Leaf):
            return node.predictions

        if node.question.match(X):
            return self.classify(X, node.true_branch)
        else:
            return self.classify(X, node.false_branch)

    def fit(self, X):
        gain, question = self.find_best_split(X)
        if gain == 0:
            return Leaf(X)

        true_X, false_X = self.partition(X, question)
        true_branch = self.fit(true_X)
        false_branch = self.fit(false_X)

        self.tree = Decision_Node(question, true_branch, false_branch)
        return self.tree

    def predict(self, X):
        y = []

        for row in X:
            for _ in self.classify(row, self.tree):
                y.append(_)

        return np.array(y)
