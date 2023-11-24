import numpy as np
import matplotlib.pyplot as plt

# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
print(np.mean(t_train))
# (b) RMSE function
v_t_train = np.full(t_train.size, np.mean(t_train))
def rmse(t, tp):
    return np.sqrt(np.mean(np.linalg.norm(t-tp)**2))
print(rmse(t_test, v_t_train))

# (c) visualization of results
plt.scatter(t_test, v_t_train)
plt.xlabel('True house prices')
plt.ylabel('Estimates')
plt.title('2D prices plot')
plt.show()