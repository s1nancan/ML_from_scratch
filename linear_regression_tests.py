import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features= 1, noise = 20, random_state=6)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= 125)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color = 'b', marker = 'o', s=25)
# plt.show()

# print(X_train.shape)
# print(y_train.shape)

# Lets try our own function and see how it does

from linear_regression import LinearRegression

regressor = LinearRegression(learn_rate=0.01)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

def mse(y_true, y_pred):
    # Mean Squared Error

    return np.mean((y_pred - y_true)**2)

mse_value = mse(y_test, y_pred)
print(mse_value)

y_pred_line = regressor.predict(X)
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color = 'b')
m2 = plt.scatter(X_test, y_test, color = 'r')
plt.plot(X,y_pred_line, color = 'green', linewidth=2, label='Prediction')
plt.show()