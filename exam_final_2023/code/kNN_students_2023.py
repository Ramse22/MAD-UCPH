import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random


def data_reading(file_name):
    train_data = pd.read_csv(file_name)
    data_numerical = train_data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']].values
    data_combined = train_data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Sex', 'ChestPainType']].values
    data_labels = train_data[['HeartDisease']].values

    print("Number of samples: %i" % data_numerical.shape[0])
    print("Number of numerical features: %i" % data_numerical.shape[1])
    print("Number of combined features: %i" % data_combined.shape[1])
    pass


class NearestNeighborRegressor:

    def __init__(self, n_neighbors=3):
        """
        Initializes the model.

        Parameters
        ----------
        n_neighbors : The number of nearest neigbhors (default 1)
        weights : weighting factors for numerical and categorical features
        """

        self.n_neighbors = n_neighbors
        self.type = 'unknown'

    def fit(self, X, t, type='numerical', weights=[1, 1]):
        """
        Fits the nearest neighbor regression model.
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of labels [n_samples]
        type: Could be 'numerical' or 'combined'
        weights: coefficients that are used to be
        """
        # .....

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

        # .....

        return predictions

    def __numericalDistance(self, p, q):
        """
        Computes the Euclidean distance between
        two points.
        """

        # .....

        return distance

    def __mixedDistance(self, p, q):
        """
        Computes the distance between
        two points via the pre-defined matrix.
        """

        # .....

        return distance

    def rmse(t, tp):
        """ Computes the RMSE for two
        input arrays 't' and 'tp'.
        """

    def accuracy(t, tp):
        """ Computes the RMSE for two
        input arrays 't' and 'tp'.
        """
