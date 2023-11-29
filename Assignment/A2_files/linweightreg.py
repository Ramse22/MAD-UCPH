import numpy as np

# NOTE: This template makes use of Python classes. If
# you are not yet familiar with this concept, you can
# find a short introduction here:
# http://introtopython.org/classes.html


class LinearRegression:
    """
    Linear regression implementation.
    """

    def __init__(self):
        pass

    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """

        X = np.array(X).reshape((len(X), -1))
        t = np.array(t).reshape((len(t), 1))

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        A = np.diagflat(np.square(t))

        self.w = np.linalg.solve(X.T @ A @ X, X.T @ A @ t)

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """
        X = np.array(X).reshape((len(X), -1))

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

        t_new = np.dot(X, self.w)
        return t_new
