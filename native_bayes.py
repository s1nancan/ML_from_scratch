## Bayes theorem
# P(y|X) = P(X|y)* P(y) / P(X)

# Naive Bayes assumes that all the features are mutually independent!
# P(y|X) = P(x1|y)*P(x2|y)...*P(y)/P(X)

# Select class with highest probability

# y = argmax_yP(y|X) = argmax[logP(x1|y) + logP(x2|y) + ... + logP(y)]
# Prior probability P(y): frequency 
# Class conditional probability:
#    P(xi|y) ~ Gaussian distribution : 1/sqrt(2*pi*var)*exp(-(x-mu)^2/2*var )


import numpy as np

class NaiveBayes:

    def fit(self, X, y):
        n_samples , n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64) 
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples) # we dont want integers.

    def predict(self, X):
        
        y_pred = [self._predict(x) for x in X]
        return y_pred


    def _predict(self,x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator / denominator






