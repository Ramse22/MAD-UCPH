import numpy as np
import pandas
import linreg
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

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:, 0], t_train)
print(model_single.w)

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train, t_train)
print(model_all.w)


# (d) evaluation of results
def rmse(t, tp):
    return np.sqrt(np.mean(np.linalg.norm(t - tp) ** 2))


t_predict = model_single.predict(X_test[:, 0])
print(rmse(t_test, t_predict))

plt.scatter(t_test, t_predict)
plt.xlabel("True house prices")
plt.ylabel("Estimates with first feature")
plt.title("2D prices plot with first feature")
plt.show()

t_predict = model_all.predict(X_test)
print(rmse(t_test, t_predict))

plt.scatter(t_test, t_predict)
plt.xlabel("True house prices")
plt.ylabel("Estimates with all features")
plt.title("2D prices plot with all features")
plt.show()
