import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import numpy as np
from collections import Counter


def dbscan(data, eps, min_pts):
    """DBSCAN clustering algorithm implemented without additional libraries."""

    def euclidean_distance(p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt(np.sum((np.array(p1) - np.array(p2)) ** 2))

    def region_query(point):
        """Find all neighbors within `eps` distance of a given point."""
        neighbors = []
        for idx, other_point in enumerate(data):
            if euclidean_distance(point, other_point) <= eps:
                neighbors.append(idx)
        return neighbors

    def expand_cluster(point_idx, neighbors, cluster_id):
        """Expand the cluster recursively from a core point."""
        clusters[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]

            if clusters[neighbor_idx] == -1:  # Previously marked as noise, now part of the cluster
                clusters[neighbor_idx] = cluster_id

            elif clusters[neighbor_idx] == 0:  # Not yet processed
                clusters[neighbor_idx] = cluster_id
                new_neighbors = region_query(data[neighbor_idx])
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)  # Expand with new core's neighbors

            i += 1

    # Initialize cluster labels (-1: noise, 0: unvisited)
    clusters = [0] * len(data)
    cluster_id = 0

    for point_idx, point in tqdm(enumerate(data), total=len(data), desc="Custom DBSCAN Progress"):
        if clusters[point_idx] != 0:  # Already visited
            continue

        neighbors = region_query(point)

        if len(neighbors) < min_pts:
            clusters[point_idx] = -1  # Mark as noise
        else:
            cluster_id += 1  # Start a new cluster
            expand_cluster(point_idx, neighbors, cluster_id)

    return clusters


def assign_cluster_labels(clusters, true_labels):
    """Assign the most frequent true label to each cluster."""
    cluster_labels = [-1] * len(clusters)  # Initialize with -1 (default as noise)

    # Get the most frequent true label for each cluster
    cluster_map = {}
    for idx, cluster in enumerate(clusters):
        if cluster != -1:  # Ignore noise points
            if cluster not in cluster_map:
                cluster_map[cluster] = []
            cluster_map[cluster].append(true_labels[idx])

    # Assign the most frequent true label to each cluster
    for cluster, points in cluster_map.items():
        most_frequent_label = Counter(points).most_common(1)[0][0]
        for idx in range(len(clusters)):
            if clusters[idx] == cluster:
                cluster_labels[idx] = most_frequent_label

    return cluster_labels


def plot_clusters(data, labels, title):
    """Visualize clustering results."""
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
    for idx, point in enumerate(data):
        plt.scatter(point[0], point[1], color=colors[labels[idx] % len(colors)] if labels[idx] != -1 else 'black', s=10)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')


# Load MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
data = mnist.data / 255.0
labels = mnist.target.astype(int)

# Use PCA to reduce dimensionality for visualization and clustering
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data)

# DBSCAN parameters
epsilon = 5.0
min_points = 10

# Subset sizes for comparison
subset_sizes = [100, 500, 1000, 2000, 5000]
custom_times = []
sklearn_times = []
custom_accuracies = []
sklearn_accuracies = []

for subset_size in subset_sizes:
    subset_data = data_2d[:subset_size]
    subset_labels = labels[:subset_size]

    # Measure time for custom implementation
    start_time = time.time()
    cluster_labels_custom = dbscan(subset_data, epsilon, min_points)
    custom_times.append(time.time() - start_time)

    # Assign labels for custom DBSCAN
    custom_pred_labels = assign_cluster_labels(cluster_labels_custom, subset_labels)
    custom_accuracy = accuracy_score(subset_labels, custom_pred_labels)
    custom_accuracies.append(custom_accuracy)

    # Measure time for scikit-learn implementation
    start_time = time.time()
    sklearn_dbscan = DBSCAN(eps=epsilon, min_samples=min_points)
    sklearn_labels = sklearn_dbscan.fit_predict(subset_data)
    sklearn_times.append(time.time() - start_time)

    # Assign labels for scikit-learn DBSCAN
    sklearn_pred_labels = assign_cluster_labels(sklearn_labels, subset_labels)
    sklearn_accuracy = accuracy_score(subset_labels, sklearn_pred_labels)
    sklearn_accuracies.append(sklearn_accuracy)

    # Print timing and accuracy results
    print(f"Subset Size: {subset_size}")
    print(f"Custom DBSCAN Time: {custom_times[-1]:.4f} seconds")
    print(f"Scikit-learn DBSCAN Time: {sklearn_times[-1]:.4f} seconds")
    print(f"Custom DBSCAN Accuracy: {custom_accuracies[-1]:.4f}")
    print(f"Scikit-learn DBSCAN Accuracy: {sklearn_accuracies[-1]:.4f}\n")

# Plot execution time comparison
plt.figure(figsize=(10, 6))
plt.plot(subset_sizes, custom_times, label="Custom DBSCAN", marker='o')
plt.plot(subset_sizes, sklearn_times, label="Scikit-learn DBSCAN", marker='s')
plt.xlabel("Subset Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Subset Size")
plt.legend()
plt.grid()
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(10, 6))
plt.plot(subset_sizes, custom_accuracies, label="Custom DBSCAN Accuracy", marker='o')
plt.plot(subset_sizes, sklearn_accuracies, label="Scikit-learn DBSCAN Accuracy", marker='s')
plt.xlabel("Subset Size")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Subset Size")
plt.legend()
plt.grid()
plt.show()

# Plot clustering results for the largest subset
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_clusters(data_2d[:subset_sizes[-1]], custom_pred_labels, "Custom DBSCAN (Largest Subset)")
plt.subplot(1, 2, 2)
plot_clusters(data_2d[:subset_sizes[-1]], sklearn_pred_labels, "Scikit-learn DBSCAN (Largest Subset)")
plt.tight_layout()
plt.show()
