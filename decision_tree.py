# At each node, find the best split value and best split threshold
# Entropy : E:-SUM(p(X)*log(p(X))) , p(x) = #x / n
# Information Gain: E(parent) - [weighted average] * E(children)

# Approach: 

# 1- Start at the top node, at each node select best split based on best information gain.
# 2- Greedy search: Loop over all features and over all thresholds (all possible feature values)
# 3- Save the best split feature and split threshold at each node.
# 4- Build tree recursively
# 5- Apply some stopping criteria to stop growing, ie, max depth, min samples at node etc.
# 6- When we have a leaf node, store the class label of the node. 

import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y) # Count number of occurrences of each value in array of non-negative ints
    ps = hist / len(y)

    return -np.sum([p*np.log2(p) for p in ps if p>0])

class Node: 
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # If we have a value (the label for the leaf node) we return it
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split = 2, max_depth = 100, n_feats = None):

        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        # grow tree 
        self.n_feats =  X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria:
        if (depth >= self.max_depth or n_labels ==1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_features, self.n_feats, replace=False)

        ## greedy search
        best_feat, best_threshold = self._best_criteria(X,y, feat_idx)
        left_idx , right_idx = self._split(X[:,best_feat],best_threshold)
        left = self._grow_tree(X[left_idx,:], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx,:], y[right_idx], depth+1)

        return Node(best_feat, best_threshold, left, right)

    def _most_common_label(self, y):
        counter = Counter(y) # similar to np.bincount
        most_common = counter.most_common(1)[0][0] # returns a tuple, value and # of occurences
        return most_common

    def _best_criteria(self, X, y, feat_idxs):
        # calculate information gain
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:,feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y,X_column, threshold)

                if gain > best_gain:
                    best_gain = gain 
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh


    def _information_gain(self, y, X_column, threshold):
        # parent E
        parent_entropy = entropy(y)
        # generate split
        left_idxs , right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) ==0:
            return 0 # meaning split didnt result anything, every elements stayed in one end
        # weighteed average of child E

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        #print('left indexes: ',left_idxs)
        #print('y values ',y[left_idxs])
        e_l , e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n)* e_l + (n_r/n)*e_r
        
        # return informatin gain
        ig = parent_entropy - child_entropy
        return ig

    def _split(self,X_columnm, split_thresh):
        left_idx = np.argwhere(X_columnm <= split_thresh).flatten()
        # return all values where value is smaller than threshold
        right_idx = np.argwhere(X_columnm > split_thresh).flatten()
        return left_idx, right_idx
        




    def predict(self, X):
        # traverse tree
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)