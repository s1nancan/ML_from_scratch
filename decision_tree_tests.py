import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from decision_tree import DecisionTree


def accuracy(y_true, y_pred):
    return np.sum(y_true==y_pred) / len(y_true)

ds = datasets.load_breast_cancer()
X= ds.data
y= ds.target
#print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

clf = DecisionTree(max_depth = 10)
clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)

print('Decision tree classification accuracy', accuracy(y_test, y_pred))

