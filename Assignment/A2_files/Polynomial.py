import numpy as np


class Polynomial:
    """
    Linear regression implementation.
    """

    def __init__(self, order):
        self.order = order
        pass

    def transform(self, X):
        """create the X matrix"""

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate([np.power(X, i) for i in range(1, self.order+1)], axis=1)
        return np.concatenate((ones, X), axis=1)

    def fit(self, X, t, l):
        """
        Fits a n order polynomial to the data
        """

        X = self.transform(np.array(X).reshape((len(X), -1)))
        t = np.array(t).reshape((len(t), 1))

        # self.w = np.linalg.inv(X.T @ X + X.shape[0] @ l) @ X.T @ t
        self.w = np.linalg.solve(X.T @ X + X.shape[0] * l, X.T @ t)

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
        X = self.transform(np.array(X).reshape((len(X), -1)))

        t_new = np.dot(X, self.w)
        return t_new
