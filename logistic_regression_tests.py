import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


bc= datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 125)


def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred)/len(y_true)
    return acc


from logistic_regression import LogisticRegression

regressor = LogisticRegression(learn_rate=0.01, n_iter=1000)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

print("Accuracy:", accuracy(y_test, y_pred))