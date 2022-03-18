import numpy as np

class LinearRegression:

    def __init__(self, learn_rate = 0.001, n_iter = 1000):
        self.learn_rate = learn_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    
    def fit(self, X,y):
        # initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # gradient descent
            y_pred = np.dot(X,self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learn_rate*dw
            self.bias -= self.learn_rate*db

    def predict(self, X):
        
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

    