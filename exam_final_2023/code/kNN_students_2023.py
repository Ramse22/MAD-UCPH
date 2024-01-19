import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random


def data_reading(file_name):
    train_data = pd.read_csv(file_name)
    data_numerical = train_data[["Age", "RestingBP", "Cholesterol", "MaxHR"]].values
    data_combined = train_data[
        ["Age", "RestingBP", "Cholesterol", "MaxHR", "Sex", "ChestPainType"]
    ].values
    data_labels = train_data[["HeartDisease"]].values

    print("Number of samples: %i" % data_numerical.shape[0])
    print("Number of numerical features: %i" % data_numerical.shape[1])
    print("Number of combined features: %i" % data_combined.shape[1])
    return data_numerical, data_combined, data_labels


class NearestNeighborRegressor:
    def __init__(self, n_neighbors):
        """
        Initializes the model.

        Parameters
        ----------
        n_neighbors : The number of nearest neigbhors (default 1)
        weights : weighting factors for numerical and categorical features
        """

        self.n_neighbors = n_neighbors
        self.type = "unknown"

    def fit(self, X, t, type="numerical", weights=[1, 1]):
        """
        Fits the nearest neighbor regression model.
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of labels [n_samples]
        type: Could be 'numerical' or 'combined'
        weights: coefficients that are used to be
        """
        self.X = X
        self.t = t
        self.weights = weights
        self.type = type

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of length n_samples
        """

        predicted_labels = []
        for i in range(len(X)):
            distance = (
                self.__numericalDistance(X[i], self.X)
                if self.type == "numerical"
                else self.__mixedDistance(X[i], self.X)
            )
            idx = np.argsort(distance)[: self.n_neighbors]
            predict_label = np.bincount(self.t[idx].T[0, :]).argmax()
            predicted_labels.append(predict_label)
        return np.array(predicted_labels)

    def __numericalDistance(self, p, q):
        """
        Computes the Euclidean distance between
        two points.
        """
        return np.linalg.norm(p - q, axis=1)

    def __mixedDistance(self, p, q):
        """
        Computes the distance between
        two points via the pre-defined matrix.
        """

        num_col = []
        cat_col = []
        for idx, val in enumerate(p):
            if type(val) in [np.float_, float]:
                num_col.append(idx)
            else:
                cat_col.append(idx)

        num_dis = self.__numericalDistance(
            p[num_col].astype(np.float64), q[:, num_col].astype(np.float64)
        )
        cat_dis = np.sum(1 - (p[cat_col] == q[:, cat_col]), axis=1)

        return self.weights[0] * num_dis + self.weights[1] * cat_dis

    @staticmethod
    def rmse(t, tp):
        """Computes the RMSE for two
        input arrays 't' and 'tp'.
        """
        return np.sqrt(np.mean((t - tp) ** 2))

    @staticmethod
    def accuracy(t, tp):
        """Computes the RMSE for two
        input arrays 't' and 'tp'.
        """
        return np.mean(t == tp)
