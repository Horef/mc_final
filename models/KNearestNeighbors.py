from sklearn.neighbors import KNeighborsRegressor

# Implementation of K-Nearest Neighbors algorithm
class KNN:
    def __init__(self, n_neighbors=6, weights='distance', metric='minkowski', n_jobs=-1):
        """
        Used to initialize the KNN model.
        :param n_neighbors: number of neighbors to consider in KNN.
        :param weights: how to weight the neighbors (uniform or distance).
        :param metric: metric to use for distance calculation (cosine or minkowski).
        :param n_jobs: number of jobs to run in parallel.
        """
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=n_jobs)

    def fit(self, X, y):
        """
        Used to train the KNN model.
        :param X: training data.
        :param y: training labels.
        :return: nothing.
        """
        self.knn.fit(X, y)

    def predict(self, X):
        """
        Used to predict the labels for the given data.
        :param X: data to predict the labels for.
        :return: predicted labels.
        """
        return self.knn.predict(X)