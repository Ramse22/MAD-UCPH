import numpy as np

# raw data
X = np.array([0.5, 1.1, -0.7, 1.5, -1.2, 0.9])
Y = np.around(2 * X / (4 - X), 2)
print(Y)
# compute means
X_mean = np.around(np.mean(X), 2)
Y_mean = np.around(np.mean(Y), 2)
mean_point = np.array([X_mean, Y_mean])
print(X_mean, Y_mean, mean_point)

# compute centered data matrix
X_cent = np.array([X - X_mean, Y - Y_mean])
print(X_cent)

# cov matrix:
Cov = (X_cent) @ (X_cent).T / len(X)
print(Cov)

# eigenvalues and vectors:
eigenvalues, eigenvectors = np.linalg.eig(Cov)
print("eigenvalues:", eigenvalues, "eigenvectors:", eigenvectors)
