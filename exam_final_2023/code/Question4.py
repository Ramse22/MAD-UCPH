from kNN_students_2023 import *

X_train_num, X_train_com, t_train = data_reading(
    "../data/heart_simplified_train_2023.csv"
)
X_test_num, X_test_com, t_test = data_reading("../data/heart_simplified_test_2023.csv")
n_neighbors = [1, 3, 5, 7, 9]

accuracies = []

for i in n_neighbors:
    kNN_classifier = NearestNeighborRegressor(i)
    kNN_classifier.fit(X_train_num, t_train)
    prediction = kNN_classifier.predict(X_test_num)
    accuracies.append(kNN_classifier.accuracy(t_test, prediction))
    print(
        f"k={i}, RMSE={kNN_classifier.rmse(t_test, prediction)} accuracy={kNN_classifier.accuracy(t_test,prediction)}"
    )

plt.scatter(n_neighbors, accuracies)
plt.xlabel("number of neighbors")
plt.ylabel("accuracy")
plt.title("accuracy of kNNTest as a function of number of neighbors")
plt.show()

Weights = [0.025, 0.5, 0.1]
kNN_classifier = NearestNeighborRegressor(5)

for i in Weights:
    kNN_classifier.fit(X_train_com, t_train, type="combined", weights=[1, i])
    prediction = kNN_classifier.predict(X_test_com)
    print(
        f"w_cat={i}, RMSE={kNN_classifier.rmse(t_test, prediction)} accuracy={kNN_classifier.accuracy(t_test,prediction)}"
    )
