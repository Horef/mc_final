import numpy as np

class Baseline:
    def __init__(self, mean=76.801748):
        self.mean = mean

    def fit(self, X, y):
        self.mean = y.mean()

    def predict(self, X):
        return np.full(X.shape[0], self.mean)