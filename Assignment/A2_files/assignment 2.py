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

#Exercise 1, b)
model_all = linweightreg.LinearRegression()
model_all.fit(X_train, t_train)
print(model_all.w)

t_predict = model_all.predict(X_test)
plt.scatter(t_test, t_predict)
plt.xlabel("True house prices")
plt.ylabel("Estimates from weighted linear regression")
plt.title("2D prices plot from weighted linear regression estimates")
plt.savefig("WeightedLinearReg.png", bbox_inches="tight")
plt.show()


