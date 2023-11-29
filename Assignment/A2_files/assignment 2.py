import numpy as np
import pandas
import linweightreg
import matplotlib.pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:, :-1], train_data[:, -1]
X_test, t_test = test_data[:, :-1], test_data[:, -1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

A = np.fill_diagonal(np.zeros((t_train.size, t_train.size)), np.square(t_train))
print(A)

model_all = linweightreg.LinearRegression()
model_all.fit(X_train, A, t_train)
print(model_all.w)
