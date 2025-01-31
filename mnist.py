import time
import unittest
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score
import numpy as np
from scipy.stats import mode
from tqdm import tqdm

def dbscan(data, eps, min_pts):
    def euclidean_distance(p1, p2):
        return sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))) ** 0.5

    def region_query(point):
        neighbors = []
        for idx, other_point in enumerate(data):
            if euclidean_distance(point, other_point) <= eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(point_idx, neighbors, cluster_id):
        clusters[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if clusters[neighbor_idx] == -1:
                clusters[neighbor_idx] = cluster_id

            elif clusters[neighbor_idx] == 0:
                clusters[neighbor_idx] = cluster_id
                new_neighbors = region_query(data[neighbor_idx])
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)

            i += 1

    clusters = [0] * len(data)
    cluster_id = 0

    for point_idx, point in enumerate(tqdm(data)):
        if clusters[point_idx] != 0:
            continue

        neighbors = region_query(point)

        if len(neighbors) < min_pts:
            clusters[point_idx] = -1
        else:
            cluster_id += 1
            expand_cluster(point_idx, neighbors, cluster_id)

    return clusters

def plot_clusters(data, labels, title):
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    for idx, point in enumerate(data):
        plt.scatter(point[0], point[1], color=colors[labels[idx] % len(colors)] if labels[idx] != -1 else 'black', s=10)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

def map_clusters_to_labels(cluster_labels, true_labels):
    unique_clusters = set(cluster_labels)
    cluster_to_label = {}

    for cluster in unique_clusters:
        if cluster == -1:
            continue

        indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        true_labels_in_cluster = true_labels[indices]

        if len(true_labels_in_cluster) > 0:
            most_common_label = mode(true_labels_in_cluster, keepdims=True).mode[0]
            cluster_to_label[cluster] = most_common_label
        else:
            cluster_to_label[cluster] = -1

    predicted_labels = [cluster_to_label[label] if label in cluster_to_label else -1 for label in cluster_labels]
    return predicted_labels

mnist = fetch_openml('mnist_784', version=1)
data = mnist.data
labels = mnist.target.astype(int)

pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

sizes = [100, 1000, 2000, 3000]
custom_times = []
sklearn_times = []
custom_metrics = []
sklearn_metrics = []

for size in sizes:
    data_points = data_2d[:size]
    true_labels = labels[:size]

    distances = np.sort(np.linalg.norm(data_points[:, np.newaxis] - data_points, axis=2), axis=1)[:, 1]
    epsilon = np.percentile(distances, 95)
    min_points = 10

    start_time = time.time()
    cluster_labels_custom = dbscan(data_points, epsilon, min_points)
    custom_time = time.time() - start_time
    custom_times.append(custom_time)

    start_time = time.time()
    sklearn_dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
    sklearn_labels = sklearn_dbscan.fit_predict(data_points)
    sklearn_time = time.time() - start_time
    sklearn_times.append(sklearn_time)

    predicted_labels_custom = map_clusters_to_labels(cluster_labels_custom, true_labels)
    predicted_labels_sklearn = map_clusters_to_labels(sklearn_labels, true_labels)

    custom_accuracy = accuracy_score(true_labels, predicted_labels_custom)
    sklearn_accuracy = accuracy_score(true_labels, predicted_labels_sklearn)

    custom_ari = adjusted_rand_score(true_labels, cluster_labels_custom)
    sklearn_ari = adjusted_rand_score(true_labels, sklearn_labels)

    custom_metrics.append((custom_accuracy, custom_ari))
    sklearn_metrics.append((sklearn_accuracy, sklearn_ari))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_clusters(data_points, cluster_labels_custom, f"Custom DBSCAN (Size: {size})")
    plt.subplot(1, 2, 2)
    plot_clusters(data_points, sklearn_labels, f"Scikit-learn DBSCAN (Size: {size})")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sizes, custom_times, label="Custom DBSCAN Time", marker='o')
plt.plot(sizes, sklearn_times, label="Scikit-learn DBSCAN Time", marker='x')
plt.xlabel("Number of Points")
plt.ylabel("Execution Time (s)")
plt.title("Execution Time vs Number of Points")
plt.legend()
plt.grid(True)
plt.show()

class TestDBSCAN(unittest.TestCase):
    def test_cluster_count(self):
        test_data = np.array([[1, 1], [2, 2], [3, 3], [10, 10], [11, 11], [12, 12]])
        clusters = dbscan(test_data, eps=2.0, min_pts=2)
        self.assertEqual(len(set(clusters)) - (1 if -1 in clusters else 0), 2)

    def test_noise_points(self):
        test_data = np.array([[1, 1], [10, 10], [20, 20]])
        clusters = dbscan(test_data, eps=1.0, min_pts=2)
        self.assertTrue(all(label == -1 for label in clusters))

    def test_single_cluster(self):
        test_data = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        clusters = dbscan(test_data, eps=2.0, min_pts=2)
        self.assertEqual(len(set(clusters)) - (1 if -1 in clusters else 0), 1)

if __name__ == "__main__":
    unittest.main()
