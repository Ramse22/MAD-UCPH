import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_data(file_name):
    data = pd.read_csv(file_name).to_numpy()
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)

    return (data - data_min) / (data_max - data_min)


# b)
class Kmeans:
    def __init__(self, k, nb_iterations=5):
        self.k = k
        self.nb_iterations = nb_iterations
        self.centroids = None
        self.labels = None
        self.cluster_sizes = None
        pass

    def fit(self, X_train):
        self.__init_centroids(X_train)
        for _ in range(self.nb_iterations):
            self.__assign_labels(X_train)

            for centroid in range(self.k):
                cluster_points = X_train[np.array(self.labels) == centroid]
                self.centroids[centroid] = np.mean(cluster_points, axis=0)

        self.cluster_sizes = [
            np.sum(np.array(self.labels) == centroid) for centroid in range(self.k)
        ]

    def __init_centroids(self, X):
        self.centroids = [X[np.random.randint(len(X))] for _ in range(self.k)]

    @staticmethod
    def __euclidean_distance(x, y):
        return np.sqrt(np.sum(np.square(x - y), axis=0))

    def __assign_labels(self, X):
        self.labels = []
        for point in range(len(X)):
            distances = []
            for centroid in self.centroids:
                distances.append(self.__euclidean_distance(X[point, :], centroid))
            self.labels.append(np.argmin(distances))

    def predict(self, X_test):
        self.__assign_labels(X_test)
        return self.labels


def recursive_HierarchicalClustering(data, k):
    kmeans = Kmeans(k[0])
    kmeans.fit(data)
    labels = kmeans.predict(data)

    if len(k) == 1:
        return data, labels, kmeans.centroids, kmeans.cluster_sizes
    else:
        data_split = [data[np.where(np.array(labels) == i)] for i in range(kmeans.k)]

        return [
            recursive_HierarchicalClustering(data_split[i], k[1:])
            for i in range(kmeans.k)
        ]


data = read_data("../data/housing.csv")
# k_means = Kmeans(5)

hierarchical_kmeans = recursive_HierarchicalClustering(data, [3, 2, 2])

data_extract = []
for i in range(3):
    for j in range(2):
        data_extract.append(
            (
                hierarchical_kmeans[i][j][0],
                hierarchical_kmeans[i][j][1],
                hierarchical_kmeans[i][j][2],
            )
        )

data_result = None
labels_result = []
centroids_result = []
for idx, i in enumerate(data_extract):
    if data_result is None:
        data_result = i[0]
    else:
        data_result = np.concatenate((data_result, i[0]))
    labels_result.extend([label + idx * 2 for label in i[1]])
    centroids_result.extend(i[2])
centroids = np.array(centroids_result)

cluster_sizes_result = []

for i, hierarchical_level in enumerate(hierarchical_kmeans):
    for j in range(2):
        cluster_sizes_result.extend(hierarchical_level[j][3])

cluster_sizes = np.array(cluster_sizes_result)
for i, size in enumerate(cluster_sizes):
    print(f"Cluster {i}: {size} points")


# c)
def pca(data):
    data_cent = data - np.mean(data)
    data_corr = 1 / len(data_cent) * np.dot(data_cent.T, data_cent)
    PCevals, PCevecs = np.linalg.eig(data_corr)
    return PCevals, PCevecs


PCevals, PCevecs = pca(data_result)

idx = PCevals.argsort()[::-1]
eigenvalues = PCevals[idx]
eigenvectors = PCevecs[:, idx]
transformed_data = np.dot(data_result, eigenvectors[:, :2])
transformed_centroid = np.dot(centroids, eigenvectors[:, :2])


# d)
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], s=20, c=labels_result)
plt.scatter(transformed_centroid[:, 0], transformed_centroid[:, 1], s=50, c="black")
plt.savefig("clustering.png")
plt.show()
