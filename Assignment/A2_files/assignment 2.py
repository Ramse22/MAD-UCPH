import numpy as np
import pandas
import linweightreg
import Polynomial
import matplotlib.pyplot as plt


def rmse(t, tp):
    return np.sqrt(np.mean((t - tp) ** 2))


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

# Exercise 1, b)
linearRegression = linweightreg.LinearRegression()
linearRegression.fit(X_train, t_train)
print(linearRegression.w)

t_predict = linearRegression.predict(X_test)
plt.scatter(t_test, t_predict)
plt.xlabel("True house prices")
plt.ylabel("Estimates from weighted linear regression")
plt.title("2D prices plot from weighted linear regression estimates")
plt.savefig("WeightedLinearReg.png", bbox_inches="tight")
plt.show()

# Exercise 2, a)

raw = np.genfromtxt("men-olympics-100.txt", delimiter=" ")

X, t = train_data[:, 0], train_data[:, 1]
# reshape both arrays to make sure that we deal with
# N-dimensional Numpy arrays
t = t.reshape((len(t), 1))
X = X.reshape((len(X), 1))
print("Shape of our data matrix: %s" % str(X.shape))
print("Shape of our target vector: %s" % str(t.shape))

lambdas = np.concatenate([np.array([0]),np.logspace(-8, 0, 100, base=10)])
computed_rmse = np.zeros(lambdas.shape)

polynomial = Polynomial.Polynomial(order=1)
for idx, lam in enumerate(lambdas):
    t_predict = np.zeros(t.shape)
    for i in range(0, X.shape[0]):
        polynomial.fit(np.delete(X, i), np.delete(t, i), lam)
        t_predict[i] = polynomial.predict(X[i])
    computed_rmse[idx] = rmse(t,t_predict)
    # print("lam=%.10f and rmse=%.10f" % (lam, computed_rmse[idx]))


plt.scatter(lambdas, computed_rmse)
plt.xlabel("lambda")
plt.ylabel("RMSE value")
plt.savefig("lambda_rmse_firstorder.png", bbox_inches="tight")
plt.show()

idx = np.argmin(computed_rmse)
polynomial.fit(X,t,lambdas[idx])
print("lam=%.10f and w0=%.10f and w1=%.10f" % (float(lambdas[idx]), polynomial.w[0,0], polynomial.w[1,0]))
polynomial.fit(X,t,lambdas[0])
print("lam=%.10f and w0=%.10f and w1=%.10f" % (float(lambdas[0]), polynomial.w[0,0], polynomial.w[1,0]))

# Exercice 2, b)

computed_rmse = np.zeros(lambdas.shape)

polynomial = Polynomial.Polynomial(order=4)
for idx, lam in enumerate(lambdas):
    t_predict = np.zeros(t.shape)
    for i in range(0, X.shape[0]):
        polynomial.fit(np.delete(X, i), np.delete(t, i), lam)
        t_predict[i] = polynomial.predict(X[i])
    computed_rmse[idx] = rmse(t,t_predict)
    # print("lam=%.10f and rmse=%.10f" % (lam, computed_rmse[idx]))



plt.scatter(lambdas, computed_rmse)
plt.xlabel("lambda")
plt.ylabel("RMSE value")
plt.savefig("lambda_rmse_fourthorder.png", bbox_inches="tight")
plt.show()

idx = np.argmin(computed_rmse)
polynomial.fit(X,t,lambdas[idx])
print("lam=%.10f and w0=%.10f and w1=%.10f and w2=%.10f and w3=%.10f and w4=%.10f" % (float(lambdas[idx]), polynomial.w[0,0], polynomial.w[1,0], polynomial.w[2,0], polynomial.w[3,0], polynomial.w[4,0]))
polynomial.fit(X,t,lambdas[0])
print("lam=%.10f and w0=%.10f and w1=%.10f and w2=%.10f and w3=%.10f and w4=%.10f" % (float(lambdas[0]), polynomial.w[0,0], polynomial.w[1,0], polynomial.w[2,0], polynomial.w[3,0], polynomial.w[4,0]))