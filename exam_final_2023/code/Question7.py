import numpy as np
import pandas as pd


def read_data(file_name):
    data = pd.read_csv(file_name).to_numpy()
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    return (data - data_min) / (data_max - data_min)


class Kmeans:
    def __init__(self, k, nb_iterations):
        self.k = k
        self.nb_iterations = nb_iterations
        self.centroids = None
        self.labels = None
        pass

    def fit(self, X_train):
        self.__init_centroids(X_train)
        for _ in range(self.nb_iterations):
            self.__assign_labels(X_train)

            for centroid in range(self.k):
                self.centroids[centroid] = np.mean(X_train[np.array(self.labels) == centroid], axis = 0)


        pass

    def __init_centroids(self, X):
        self.centroids = [X[np.random.randint(len(X))] for _ in range(self.k)]

    @staticmethod
    def __euclidean_distance(x,y):
        return np.sqrt(np.sum(np.square(x - y), axis=0))

    def __assign_labels(self, X):
        self.labels = []
        for point in range(len(X)):
            distances = []
            for centroid in self.centroids:
                distances.append(self.__euclidean_distance(X[point, :], centroid))
            self.labels.append(np.argmin(distances))

    def predict(self, X_test):
        self.__assign_labels(X_test)
        return self.labels


data = read_data("../data/housing.csv")
k_means = Kmeans(5)
