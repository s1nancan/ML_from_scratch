import numpy as np


class LogisticRegression:

    def __init__(self, learn_rate=0.001, n_iter=1000):

        self.learn_rate = learn_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # initialize parameters

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent

        for _ in range(self.n_iter):

            linear_model = np.dot(X,self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learn_rate*dw
            self.bias -= self.learn_rate*db

    def predict(self, X):
        
        # Do a linear prediction and apply sigmoid. 
        y_pred_linear = np.dot(X,self.weights) + self.bias
        y_pred = self._sigmoid(y_pred_linear)
        # User 0.5 as a threshold to assign a class
        y_pred_class = [1 if i > 0.5 else 0 for i in y_pred]
        
        return y_pred_class



    def _sigmoid(self, x):
        return 1 / (1+np.exp(-x))